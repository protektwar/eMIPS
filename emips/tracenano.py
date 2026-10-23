"""The views of a nanoMIPS instruction (PLAN M23 phase 4a; emips.trace).

:mod:`emips.trace` records a nanoMIPS instruction as it does a MIPS32 one
(the state before and after, the memory, cache and TLB logs); this module
builds the parts that depend on the encoding:

* **decode** — one field per token of the format's encoding diagram
  (MD01247 ch. 3, the row's template in :mod:`emips.nanomips`): the major
  opcode with the pool it selects (Tables B.2 / B.3), the other fixed
  bits, the operand fields with the arguments the format's decode
  pseudocode makes of them (``rt3 = 5: rt = gpr3(rt3) = $a1``), the bits
  the hardware ignores (A.4.3);
* **ALU**, **address**, **branch** — from the decoded arguments of the
  instruction page (``m._nm.f``), the register values read before it ran
  and what it changed.

Nothing here reads the live machine: :func:`emips.trace.begin` keeps the
decoded instruction in the frozen state.
"""

from __future__ import annotations

from typing import Any, Callable

from . import nanomips as NM
from .simnano import uses
from .syntax import gpr_name

M32 = 0xFFFFFFFF

#: the major opcode (instruction bits 15:10 of the first halfword) -> the
#: pool or format it selects: MD01247 Table B.2 (P32, bit 12 = 0) and
#: Table B.3 (P16, bit 12 = 1); "*" = reserved
_P32 = ("P.ADDIU ADDIUPC[32] MOVE.BALC * P32A * P.BAL * P.GP.W P.GP.BH P.J * "
        "P48I * * * P.U12 P.LS.U12 P.BR1 * *(CP1) P.LS.S9 P.BR2 * *(MIPS64) * P.BRI * "
        "P.LUI * * *").split()
_P16 = ("P16.MV LW[16] BC[16] P16.SR P16.SHIFT LW[SP] BALC[16] P16.4X4 P16C LW[GP16] * "
        "P16.LB P16.A1 LW[4X4] * P16.LH P16.A2 SW[16] BEQZC[16] * P16.ADDU SW[SP] "
        "BNEZC[16] MOVEP LI[16] SW[GP16] P16.BR * ANDI[16] SW[4X4] * MOVEP[REV]").split()
MAJOR = {hi << 3 | p16 << 2 | lo: (_P16 if p16 else _P32)[hi * 4 + lo]
         for hi in range(8) for p16 in (0, 1) for lo in range(4)}

#: the decode functions' meaning (emips.nanomips._FUNCS)
_REGISTER_FUNCS = frozenset(("gpr3", "gpr3s", "gpr1", "gpr2a", "gpr2b", "gpr4", "gpr4z",
                             "rt1"))
#: decoded arguments that are GPR numbers (the Operations' names)
_REGISTERS = frozenset(("rt", "rs", "rd", "ru", "dst", "src", "dst1", "dst2", "src1",
                        "src2"))


def _sx(v: int, bits: int) -> int:
    return v - (1 << bits) if v >> (bits - 1) & 1 else v


def _derived(row: NM.Row) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """For each encoded field, the arguments the decode steps make of it
    (in step order, transitively), and each argument's steps as the
    pseudocode writes them (``sx(s, 15)``, ``gpr3(rt3)``)."""
    roots: dict[str, set[str]] = {}
    made: dict[str, list[str]] = {}
    exprs: dict[str, list[str]] = {}
    for target, fn, arg, const in row.steps:
        src: set[str] = set()
        for a in (arg, const):
            if isinstance(a, str):
                src |= roots.get(a, {a})
        if not src:                                 # a constant (rs = 29): no field
            continue
        roots[target] = src
        args = ", ".join(str(a) for a in (arg, const) if a is not None)
        exprs.setdefault(target, []).append(f"{fn}({args})" if fn else str(arg))
        for r in src:
            made.setdefault(r, [])
            if target not in made[r]:
                made[r].append(target)
    return made, exprs


def _num(v: int) -> str:
    return str(v) if -10 < v < 10 else f"{v} (0x{v & M32:x})"


def _raw_args(row: NM.Row, raw: int) -> dict[str, int]:
    """The encoded fields' values before any decode step."""
    out = {}
    for name, pieces in row.fields:
        v = 0
        for shift, width, at in pieces:
            v |= ((raw >> shift) & ((1 << width) - 1)) << at
        out[name] = v
    return out


def fields(i: NM.NM, abi: Any = "p32") -> list:
    """The :class:`emips.trace.Field` of every token of the format's
    encoding, most significant first (bit numbers of the whole 16 / 32 /
    48-bit encoding, MD01247 A.2); an operand field's meaning is its value
    and, on its last token, what the format's decode pseudocode makes of
    it."""
    from .trace import Field
    row, raw, f = i.row, i.raw, i.f
    size = row.size * 8
    toks = row.template.split()
    made, exprs = _derived(row)
    plain = _raw_args(row, raw)
    last_fixed = max((k for k, t in enumerate(toks) if t[0] in "01"), default=-1)
    last_of = {}
    for k, t in enumerate(toks):
        m = NM._TOKEN.match(t)
        if not m.group(1):
            last_of[m.group(2) or m.group(4)] = k
    out: list[Field] = []
    pos = size
    for k, t in enumerate(toks):
        m = NM._TOKEN.match(t)
        if m.group(1):
            width = len(m.group(1))
        elif m.group(2):
            width = int(m.group(3))
        else:
            width = int(m.group(5)) - int(m.group(6) or m.group(5)) + 1
        pos -= width
        hi, lo = pos + width - 1, pos
        value = (raw >> lo) & ((1 << width) - 1)
        bits = format(value, f"0{width}b")
        if m.group(1):                                  # fixed bits
            if k == 0 and width == 6:
                pool = MAJOR[value]
                name, meaning = "major", pool if pool == row.form else f"the {pool} pool"
            else:
                name, meaning = "op", row.form if k == last_fixed else ""
            out.append(Field(name, hi, lo, bits, value, meaning))
            continue
        arg = m.group(2) or m.group(4)
        if arg == "x":
            out.append(Field("x", hi, lo, bits, value, "ignored by the hardware "
                                                         "(MD01247 A.4.3)"))
            continue
        parts, derived = [], []
        if m.group(4):                                  # a slice of the argument
            parts.append((f"bit {m.group(5)}" if m.group(6) is None else
                          f"bits {m.group(5)}:{m.group(6)}") + f" of {arg}")
        if last_of[arg] == k:
            whole = plain[arg]
            if arg in _REGISTERS and arg not in exprs:
                parts.append(gpr_name(whole, abi))
            else:
                parts.append(f"{arg} = {_num(whole)}")
            for x in made.get(arg, []):
                shown = gpr_name(f[x], abi) if x in _REGISTERS else _num(f[x])
                derived.append(f"{x} = {', '.join(exprs[x])} = {shown}")
        out.append(Field(arg, hi, lo, bits, value, "; ".join([", ".join(parts)] + derived)))
    return out


# ------------------------------------------------------------------ the ALU

#: the ALU operation of each page: (expression, a, b); an operand is
#: "r:KEY" (the GPR the decoded argument KEY names), "i:KEY" (the decoded
#: immediate KEY), "pc" (the address of the next instruction) or None; the
#: expression may be a function of the decoded arguments
_OP = Callable[[dict], str] | str
_ALU: dict[str, tuple[_OP, str | None, str | None]] = {
    "ADD": ("a + b (trap on overflow)", "r:rs", "r:rt"),
    "ADDU": ("a + b", "r:src1", "r:src2"),
    "SUB": ("a - b (trap on overflow)", "r:rs", "r:rt"),
    "SUBU": ("a - b", "r:rs", "r:rt"),
    "SOV": ("a + b overflows 32 bits ? 1 : 0", "r:rs", "r:rt"),
    "AND": ("a & b", "r:rs", "r:rt"), "OR": ("a | b", "r:rs", "r:rt"),
    "XOR": ("a ^ b", "r:rs", "r:rt"), "NOR": ("~(a | b)", "r:rs", "r:rt"),
    "SLT": ("a < b ? 1 : 0 (signed)", "r:rs", "r:rt"),
    "SLTU": ("a < b ? 1 : 0 (unsigned)", "r:rs", "r:rt"),
    "SLLV": ("a << (b & 31)", "r:rs", "r:rt"),
    "SRLV": ("a >> (b & 31) (logical)", "r:rs", "r:rt"),
    "SRAV": ("a >> (b & 31) (arithmetic)", "r:rs", "r:rt"),
    "ROTRV": ("a rotated right by (b & 31)", "r:rs", "r:rt"),
    "MUL": ("a * b (low 32 bits)", "r:src1", "r:src2"),
    "MUH": ("a * b (high 32 bits, signed)", "r:rs", "r:rt"),
    "MULU": ("a * b (low 32 bits, unsigned)", "r:rs", "r:rt"),
    "MUHU": ("a * b (high 32 bits, unsigned)", "r:rs", "r:rt"),
    "DIV": ("a / b (signed)", "r:rs", "r:rt"), "DIVU": ("a / b (unsigned)", "r:rs", "r:rt"),
    "MOD": ("a % b (signed)", "r:rs", "r:rt"), "MODU": ("a % b (unsigned)", "r:rs", "r:rt"),
    "MOVN": ("b != 0 ? a : rd", "r:rs", "r:rt"), "MOVZ": ("b == 0 ? a : rd", "r:rs", "r:rt"),
    "NOT": ("~a", "r:rs", None),
    "CLO": ("count leading ones of a", "r:rs", None),
    "CLZ": ("count leading zeros of a", "r:rs", None),
    "SEB": ("sign-extend byte of a", "r:rs", None),
    "SEH": ("sign-extend halfword of a", "r:rs", None),
    "EXT": (lambda f: f"extract {f['size']} bits of a from bit {f['pos']}", "r:rs", None),
    "INS": (lambda f: f"insert the low {f['size']} bits of a into rt at bit {f['pos']}",
            "r:rs", "r:rt"),
    "EXTW": (lambda f: f"(a:b) >> {f['shift']} (the low 32 bits)", "r:rt", "r:rs"),
    "LSA": (lambda f: f"(a << {f['u2']}) + b", "r:rs", "r:rt"),
    "ROTX": (lambda f: f"rotate / exchange the bits of a (shift {f['shift']}, shiftx "
                       f"{f['shiftx']}, stripe {f['stripe']})", "r:rs", None),
    **{n: (f"CRC-32{'C' if 'C' in n[5:] else ''} of the low {w} bits of b into a "
           f"(reflected, polynomial 0x{p:08x})", "r:rt", "r:rs")
       for n, w, p in (("CRC32B", 8, 0xEDB88320), ("CRC32H", 16, 0xEDB88320),
                       ("CRC32W", 32, 0xEDB88320), ("CRC32CB", 8, 0x82F63B78),
                       ("CRC32CH", 16, 0x82F63B78), ("CRC32CW", 32, 0x82F63B78))},
    "ADDIU": ("a + b", "r:rs", "i:imm"),
    "ADDIUPC": ("a + b (PC-relative)", "pc", "i:s"),
    "ALUIPC": ("(a + b) & ~0xfff (PC-relative)", "pc", "i:offset"),
    "ANDI": ("a & b", "r:rs", "i:u"), "ORI": ("a | b", "r:rs", "i:u"),
    "XORI": ("a ^ b", "r:rs", "i:u"),
    "SLTI": ("a < b ? 1 : 0 (signed)", "r:rs", "i:u"),
    "SLTIU": ("a < b ? 1 : 0 (unsigned)", "r:rs", "i:u"),
    "SEQI": ("a == b ? 1 : 0", "r:rs", "i:u"),
    "LI": ("b (the immediate)", None, "i:s"),
    "LUI": ("b (the immediate: bits 31:12)", None, "i:imm"),
    "SLL": ("a << b", "r:rs", "i:shift"), "SRL": ("a >> b (logical)", "r:rs", "i:shift"),
    "SRA": ("a >> b (arithmetic)", "r:rs", "i:shift"),
    "ROTR": ("a rotated right by b", "r:rs", "i:shift"),
}

#: loads / stores: page -> (bytes, access); the address is GPR[rs] + offset
#: unless the page says otherwise (_INDEXED, _PC_RELATIVE, SAVE / RESTORE)
_MEMORY = {
    **dict.fromkeys(("LB", "LBU", "LBE", "LBUE", "LBX", "LBUX"), (1, "load")),
    **dict.fromkeys(("LH", "LHU", "LHE", "LHUE", "UALH", "LHX", "LHUX", "LHXS", "LHUXS"),
                    (2, "load")),
    **dict.fromkeys(("LW", "LWE", "LWX", "LWXS", "LWPC", "LWM", "UALWM", "RESTORE", "LL"),
                    (4, "load")),
    **dict.fromkeys(("SB", "SBE", "SBX"), (1, "store")),
    **dict.fromkeys(("SH", "SHE", "UASH", "SHX", "SHXS"), (2, "store")),
    **dict.fromkeys(("SW", "SWE", "SWX", "SWXS", "SWPC", "SWM", "UASWM", "SAVE", "SC"),
                    (4, "store")),
}
#: indexed: (GPR[rs] << scale) + GPR[rt]
_INDEXED = {"LBX": 0, "LBUX": 0, "LHX": 0, "LHUX": 0, "LWX": 0, "LHXS": 1, "LHUXS": 1,
            "LWXS": 2, "SBX": 0, "SHX": 0, "SWX": 0, "SHXS": 1, "SWXS": 2}
#: accesses that may be misaligned (UALH ...; the I7200 and eMIPS allow the
#: others too: MD01247 4.10 — only the atomic and SAVE / RESTORE ones are not)
_UNALIGNED = frozenset(("UALH", "UASH", "UALWM", "UASWM"))
_ALIGNED = frozenset(("LL", "SC", "SAVE", "RESTORE"))


def _reg(st: Any, n: int, sources: dict | None, name: str = "a"):
    from .trace import Operand
    src = (sources or {}).get(("r", n), "register file")
    return Operand(name, st.read_reg(n), f"{gpr_name(n, st.abi)} ({src})")


def _operand(spec: str | None, st: Any, i: NM.NM, pc: int, sources: dict | None, name: str):
    from .trace import Operand
    if spec is None:
        return None
    if spec == "pc":
        return Operand(name, (pc + i.size) & M32, "the address of the next instruction")
    kind, key = spec.split(":")
    if kind == "r":
        return _reg(st, i.f[key], sources, name)
    return Operand(name, i.f[key] & M32, f"immediate ({key} = {i.f[key]}, decoded)")


def operands(v: Any, st: Any, i: NM.NM, sources: dict | None) -> None:
    """The ALU view and the address view of the instruction ``i`` at
    ``v.pc``."""
    from .trace import AddressView, AluView, Operand, _segment
    page, f, pc = i.row.name, i.f, v.pc
    spec = _ALU.get(page)
    if spec is not None:
        op, a, b = spec
        v.alu = AluView(op(f) if callable(op) else op, _operand(a, st, i, pc, sources, "a"),
                        _operand(b, st, i, pc, sources, "b"), None, None)
    mem = _MEMORY.get(page)
    if mem is None:
        return
    size, access = mem
    if page == "LL" and f["nbytes"] == 8 or page == "SC" and f["nbytes"] == 8:
        size = 8
    if page in _INDEXED:
        scale = _INDEXED[page]
        base = _reg(st, f["rs"], sources, "base")
        if scale:
            base = Operand("base", (base.value << scale) & M32, f"{base.source} << {scale}")
        off = st.read_reg(f["rt"])
        how = f"index {gpr_name(f['rt'], st.abi)}"
    elif page in ("LWPC", "SWPC"):
        base = Operand("base", (pc + i.size) & M32, "the address of the next instruction")
        off, how = _sx(f["offset"] & M32, 32), "offset (decoded)"
    elif page == "SAVE":
        base, off, how = _reg(st, 29, sources, "base"), -4, "the first register's slot"
    elif page == "RESTORE":
        base, off = _reg(st, 29, sources, "base"), f["u"] - 4
        how = "the first register's slot"
    else:
        base = _reg(st, f["rs"], sources, "base")
        off, how = f["offset"], "offset (decoded)"
    addr = (base.value + off) & M32
    seg, priv, mapped = _segment(st, addr)
    aligned = page in _UNALIGNED or addr % size == 0
    if page not in _ALIGNED and page not in _UNALIGNED:
        aligned = addr % size == 0
    v.address = AddressView(base, off, addr, size, access, seg, priv, mapped, False, aligned)
    v.alu = AluView("a + b (effective address)", base, Operand("b", off & M32, how), addr,
                    addr == 0)


def result_register(i: NM.NM) -> int | None:
    """The GPR an ALU instruction writes (the ALU view's result)."""
    writes = uses(i)[1]
    return writes[0] if writes else None


# --------------------------------------------------------------- branches

def _cond(page: str, f: dict, r: Callable[[int], str]) -> str:
    rel = {"BEQC": "==", "BNEC": "!=", "BGEC": ">= (signed)", "BLTC": "< (signed)",
           "BGEUC": ">= (unsigned)", "BLTUC": "< (unsigned)"}
    if page in rel:
        return f"{r(f['rs'])} {rel[page]} {r(f['rt'])}"
    imm = {"BEQIC": "==", "BNEIC": "!=", "BGEIC": ">= (signed)", "BLTIC": "< (signed)",
           "BGEIUC": ">= (unsigned)", "BLTIUC": "< (unsigned)"}
    if page in imm:
        return f"{r(f['rt'])} {imm[page]} {f['u']}"
    if page in ("BEQZC", "BNEZC"):
        return f"{r(f['rt'])} {'==' if page == 'BEQZC' else '!='} 0"
    if page in ("BBEQZC", "BBNEZC"):
        return f"bit {f['bit']} of {r(f['rt'])} == {0 if page == 'BBEQZC' else 1}"
    if page in ("BALRSC", "BRSC"):
        return f"always (the next instruction + {r(f['rs'])} << 1)"
    if page in ("JALRC", "JALRC.HB", "JRC"):
        return "always (jump to a register's address)"
    return "always"


def branch(v: Any, st: Any, i: NM.NM) -> None:
    """The branch view: the condition, taken (the instruction redirected the
    fetch: no delay slot, no forbidden slot), the target, the link."""
    from .trace import BranchView
    fam = i.row.name
    if v.exception or fam not in _BRANCHES:
        return
    r = lambda n: gpr_name(n, st.abi)           # noqa: E731
    target = st.redirect
    link = next((new for k, old, new in v.reg_writes if k == "$ra"), None)
    v.branch = BranchView(_cond(fam, i.f, r) + " (no delay slot)", target is not None,
                          target, link=link)


_BRANCHES = frozenset(("BC", "BALC", "BEQC", "BNEC", "BGEC", "BLTC", "BGEUC", "BLTUC",
                       "BEQIC", "BNEIC", "BGEIC", "BLTIC", "BGEIUC", "BLTIUC", "BEQZC",
                       "BNEZC", "BBEQZC", "BBNEZC", "BALRSC", "BRSC", "JALRC", "JALRC.HB",
                       "JRC", "MOVE.BALC"))
