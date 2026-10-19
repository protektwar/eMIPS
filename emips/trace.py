"""What happens inside one instruction (PLAN M6).

:func:`begin` / :func:`finish` wrap the real executor of one instruction
(:meth:`emips.sim.Machine._traced`) and build an :class:`InstrView`:

* **decode** — every bit field of the word (fixed fields named by their
  standard position: ``op rs rt rd sa funct``; operand fields named by
  their role) with bits, value and meaning (``$t1``, ``-4``, ``ADD``);
* **ALU** — the operation, its two inputs with their *source* (register
  file, forwarded from EX/MEM or MEM/WB in the pipeline, sign- or
  zero-extended immediate, shift amount) and the result, with the zero and
  overflow outputs;
* **address** — base + sign-extended offset, the effective (virtual)
  address, its segment (kuseg / kseg0 / kseg1 / kseg2 / kseg3) with the
  privilege it needs, alignment, and — observed from the access — the
  physical address, the cacheability attribute and how the MMU got them;
* **translations** — with a TLB or BAT MMU, every translation of the
  fetch and of the instruction (:class:`emips.tlb.TLBAccess`: micro-TLB
  hit or miss, the entry, VA -> PFN -> PA or the TLB exception) and what
  TLBWI / TLBWR / TLBP / TLBR / TLBINV did;
* **branch** — the condition evaluated, taken or not, the target;
* **effects** — registers, HI/LO, FPRs, FCSR, CP0 and memory as they
  changed, memory reads with their values, and the exception (if any).

The view is *observed* around the executor (operands read before, state
compared after), so it can never disagree with what the instruction did.
Nothing here changes machine state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import cp0 as C
from .disasm import disassemble
from .isa import ISA, Decode
from .mmu import cca_name, segment
from .syntax import KINDS, gpr_name, kind_of, operand_spec

M32 = 0xFFFFFFFF


@dataclass
class Field:
    name: str
    msb: int                 # bit numbers, 31 = MSB
    lsb: int
    bits: str
    value: int
    meaning: str


@dataclass
class Operand:
    name: str
    value: int
    source: str


@dataclass
class AluView:
    op: str                  # e.g. "a + b"
    a: Operand | None
    b: Operand | None
    result: int | None
    zero: bool | None
    overflow: bool = False


@dataclass
class AddressView:
    base: Operand
    offset: int
    address: int
    size: int
    access: str              # "load" / "store"
    segment: str
    privilege: str
    mapped: bool
    cached: bool             # cacheability attribute is not "uncached"
    aligned: bool
    physical: int | None = None      # None: no access (exception)
    cca: int | None = None           # cacheability and coherency attribute
    how: str = ""                    # the MMU rule that translated it


@dataclass
class BranchView:
    condition: str
    taken: bool
    target: int | None
    likely: bool = False
    link: int | None = None


@dataclass
class InstrView:
    pc: int
    word: int
    text: str
    name: str
    format: str
    fields: list[Field]
    in_delay_slot: bool = False
    alu: AluView | None = None
    address: AddressView | None = None
    branch: BranchView | None = None
    reg_writes: list[tuple[str, int, int]] = field(default_factory=list)
    mem_reads: list[tuple[int, int, int]] = field(default_factory=list)
    mem_writes: list[tuple[int, int, int, int]] = field(default_factory=list)
    exception: str | None = None
    #: the cache accesses of the instruction's fetch and of its own
    #: memory accesses (emips.cache.CacheAccess; machines with caches)
    fetch: list = field(default_factory=list)
    cache: list = field(default_factory=list)
    #: the translations (emips.tlb.TLBAccess) of the fetch and of the
    #: instruction (its accesses, TLBWI / TLBWR / TLBP / TLBR ...): a TLB
    #: or BAT MMU
    fetch_tlb: list = field(default_factory=list)
    tlb: list = field(default_factory=list)
    #: MIPS16e: the 16-bit instruction (what the fields of a recoded one
    #: show is its MIPS32 expansion, emips.mips16.expand)
    compressed: str = ""
    #: ... its text alone (what a listing shows at that address)
    source_text: str = ""
    _before: dict[str, Any] = field(default_factory=dict, repr=False)


# ------------------------------------------------------------------ decode

_FIXED_SEGMENTS = ((26, 31, "op"), (21, 25, "rs"), (16, 20, "rt"),
                   (11, 15, "rd"), (6, 10, "sa"), (0, 5, "funct"))
_OPCODE_NAMES = {0x00: "SPECIAL", 0x01: "REGIMM", 0x10: "COP0", 0x11: "COP1",
                 0x12: "COP2", 0x13: "COP1X", 0x1C: "SPECIAL2", 0x1F: "SPECIAL3"}


def instruction_format(word: int) -> str:
    op = word >> 26
    if op in (0x00, 0x1C, 0x1F):
        return "R"
    if op in (0x02, 0x03, 0x1D):
        return "J"
    if op == 0x11:
        return "FR" if (word >> 21) & 31 >= 0x10 else "FI"
    if op in (0x10, 0x12, 0x13):
        return "COP"
    return "I"


_FIELDS: dict[tuple[int, str], list[Field]] = {}


def decode_fields(dec: Decode) -> list[Field]:
    """Bit fields of the word, MSB first: fixed runs split at the standard
    field boundaries, operand fields named by their role (cached: they
    depend on the word and its row only)."""
    key = (dec.word, dec.name)
    hit = _FIELDS.get(key)
    if hit is None:
        if len(_FIELDS) >= 1 << 16:
            _FIELDS.clear()
        hit = _FIELDS[key] = _decode_fields(dec)
    return list(hit)


def _decode_fields(dec: Decode) -> list[Field]:
    e, w = dec.entry, dec.word
    mask = e["decode"]["mask"]
    owner: dict[int, str] = {}
    for item in operand_spec(e):
        kind = kind_of(item)[0]
        for lsb, width in KINDS[kind].fields:
            for b in range(lsb, lsb + width):
                owner[b] = _part_name(kind, lsb)
    out: list[Field] = []
    bit = 31
    while bit >= 0:
        hi = lo = bit
        if mask >> bit & 1:              # fixed run, cut at field boundaries
            seg_lo, _, name = next(s for s in _FIXED_SEGMENTS if s[0] <= bit <= s[1])
            while lo - 1 >= seg_lo and mask >> (lo - 1) & 1:
                lo -= 1
        else:                            # operand field
            name = owner.get(bit, "?")
            while lo - 1 >= 0 and not (mask >> (lo - 1) & 1) \
                    and owner.get(lo - 1, "?") == name:
                lo -= 1
        value = (w >> lo) & ((1 << (hi - lo + 1)) - 1)
        bits = format(value, f"0{hi - lo + 1}b")
        out.append(Field(name, hi, lo, bits, value,
                         _meaning(name, value, hi - lo + 1, dec)))
        bit = lo - 1
    return out


def _part_name(kind: str, lsb: int) -> str:
    if kind in ("mem", "mem9"):
        return "base" if lsb == 21 else "offset"
    if kind == "idx":
        return "base" if lsb == 21 else "index"
    if kind in ("pos_ext", "pos_ins"):
        return "lsb" if lsb == 6 else ("msbd" if kind == "pos_ext" else "msb")
    return {"br": "offset", "jt": "instr_index", "uimm": "immediate",
            "imm": "immediate"}.get(kind, kind)


def _sx(v: int, bits: int) -> int:
    return v - (1 << bits) if v >> (bits - 1) & 1 else v


def _meaning(name: str, v: int, width: int, dec: Decode) -> str:
    if name in ("rs", "rt", "rd", "base", "index") and width == 5:
        return gpr_name(v)
    if name in ("fs", "ft", "fd", "fr"):
        return f"$f{v}"
    if name == "op":
        return _OPCODE_NAMES.get(v, dec.name)
    if name == "funct":
        return dec.name
    if name in ("offset",) and width == 16:
        return f"{_sx(v, 16)}" + (" (x4, PC-relative)" if dec.entry["family"]
                                  in ("branch", "branch_fpu") else "")
    if name == "offset" and width == 9:
        return str(_sx(v, 9))
    if name == "immediate":
        return str(v if dec.name in ("ANDI", "ORI", "XORI", "LUI") else _sx(v, 16))
    if name == "instr_index":
        return f"target = (PC+4)[31:28] || 0x{v:07x} << 2"
    if name == "sa" and dec.entry["family"] == "shift":
        return f"shift amount {v}"
    return str(v)


# ----------------------------------------------------------------- the ALU

#: ALU function shown for each instruction: (expression, b is immediate?)
_ALU_OPS: dict[str, str] = {
    "ADD": "a + b (trap on overflow)", "ADDU": "a + b", "SUB": "a - b (trap on overflow)",
    "SUBU": "a - b", "AND": "a & b", "OR": "a | b", "XOR": "a ^ b",
    "NOR": "~(a | b)", "SLT": "a < b ? 1 : 0 (signed)",
    "SLTU": "a < b ? 1 : 0 (unsigned)", "ADDI": "a + b (trap on overflow)",
    "ADDIU": "a + b", "ANDI": "a & b", "ORI": "a | b", "XORI": "a ^ b",
    "SLTI": "a < b ? 1 : 0 (signed)", "SLTIU": "a < b ? 1 : 0 (unsigned)",
    "LUI": "b << 16", "SLL": "a << b", "SRL": "a >> b (logical)",
    "SRA": "a >> b (arithmetic)", "ROTR": "a rotated right by b",
    "SLLV": "a << (b & 31)", "SRLV": "a >> (b & 31) (logical)",
    "SRAV": "a >> (b & 31) (arithmetic)", "ROTRV": "a rotated right by (b & 31)",
    "MUL": "a * b (low 32 bits)", "MULT": "HI:LO = a * b (signed)",
    "MULTU": "HI:LO = a * b (unsigned)", "DIV": "LO = a / b, HI = a % b (signed)",
    "DIVU": "LO = a / b, HI = a % b (unsigned)", "MADD": "HI:LO += a * b",
    "MADDU": "HI:LO += a * b (unsigned)", "MSUB": "HI:LO -= a * b",
    "MSUBU": "HI:LO -= a * b (unsigned)", "CLZ": "count leading zeros of a",
    "CLO": "count leading ones of a", "MOVZ": "b == 0 ? a : rd",
    "MOVN": "b != 0 ? a : rd", "SEB": "sign-extend byte of a",
    "SEH": "sign-extend halfword of a", "WSBH": "swap bytes in halfwords of a",
    "EXT": "extract bit field of a", "INS": "insert bit field of a into rt",
}

_ZEXT_IMM = {"ANDI", "ORI", "XORI", "LUI"}
_SHIFT_IMM = {"SLL", "SRL", "SRA", "ROTR"}
_SHIFT_VAR = {"SLLV", "SRLV", "SRAV", "ROTRV"}


def _reg_operand(m, name: str, n: int, sources: dict | None) -> Operand:
    src = (sources or {}).get(("r", n), "register file")
    return Operand(name, m.read_reg(n), f"{gpr_name(n)} ({src})")


def _segment(m, addr: int) -> tuple[str, str, bool]:
    """(segment, privilege, mapped) of a virtual address."""
    seg = segment(addr)
    if seg == "kuseg":
        return seg, "user", True
    if seg == "kseg2" and m.mmu.supervisor:
        return "kseg2 (sseg)", "supervisor", True
    return seg, "kernel", seg in ("kseg2", "kseg3")


_SIZES = {"LB": 1, "LBU": 1, "SB": 1, "LH": 2, "LHU": 2, "SH": 2, "LW": 4,
          "SW": 4, "LL": 4, "SC": 4, "LWL": 4, "LWR": 4, "SWL": 4, "SWR": 4,
          "LWC1": 4, "SWC1": 4, "LDC1": 8, "SDC1": 8, "LWXC1": 4,
          "SWXC1": 4, "LDXC1": 8, "SDXC1": 8, "LUXC1": 8, "SUXC1": 8}


# ------------------------------------------------------------- begin/finish

def _compressed_text(m, i, pc: int) -> str:
    """The disassembly of the MIPS16e / microMIPS instruction ``i``."""
    if m.compressed == "micromips":
        from . import micromips
        return micromips.text(i, pc, release=m.release)
    from . import mips16
    return mips16.text(i, pc)


def _compressed(m, pc: int) -> tuple[str, str, int, int] | None:
    """(the ISA's name, the text, the encoding, its size) of the MIPS16e /
    microMIPS instruction ``m`` runs, or None in MIPS32 mode."""
    if not getattr(m, "isa_mode", 0):
        return None
    if m.compressed == "micromips":
        i = m._mm
        if i is None:
            return None
        from . import micromips
        return "microMIPS", micromips.text(i, pc, release=m.release), i.raw, i.size
    i = getattr(m, "_m16", None)
    if i is None:
        return None
    from . import mips16
    return "MIPS16e", mips16.text(i, pc), i.raw, i.size


def begin(m, pc: int, word: int, dec: Decode, in_slot: bool = False,
          sources: dict | None = None) -> InstrView:
    """Build the pre-execution part of the view (fields, operands,
    address) and snapshot the state; call :func:`finish` afterwards."""
    name, fam = dec.name, dec.entry["family"]
    c = _compressed(m, pc)
    if dec.entry.get("isa") in ("mips16e", "micromips"):    # no MIPS32 twin: its own view
        label, text, raw, size = c
        v = InstrView(pc, word, text, name, label, [], in_slot)
        v.compressed = f"{label} {text} (0x{raw:0{2 * size}x})"
        v.source_text = v.text
        v._before = _snapshot(m)
        m._mem_log = []
        return v
    isa = m.isa if c is None else _twin_isa(m)
    v = InstrView(pc, word, disassemble(word, pc, isa=isa), name,
                  instruction_format(word),
                  decode_fields(dec), in_slot)
    if c is not None:
        label, text, raw, size = c
        v.compressed = f"{label} {text} (0x{raw:0{2 * size}x}), recoded as this MIPS32 instruction"
        v.source_text = text
    rs, rt, rd, sa = (word >> 21) & 31, (word >> 16) & 31, (word >> 11) & 31, (word >> 6) & 31
    imm = word & 0xFFFF
    if name in _ALU_OPS:
        if name in _SHIFT_IMM:
            a, b = _reg_operand(m, "a", rt, sources), Operand("b", sa, "shift amount (sa field)")
        elif name in _SHIFT_VAR:
            a, b = _reg_operand(m, "a", rt, sources), _reg_operand(m, "b", rs, sources)
        elif fam == "imm":
            val = imm if name in _ZEXT_IMM else _sx(imm, 16) & M32
            how = "zero-extended" if name in _ZEXT_IMM else "sign-extended"
            a = None if name == "LUI" else _reg_operand(m, "a", rs, sources)
            b = Operand("b", val, f"immediate 0x{imm:04x} ({how})")
        elif name in ("CLZ", "CLO", "SEB", "SEH", "WSBH"):
            src = rs if name in ("CLZ", "CLO") else rt
            a, b = _reg_operand(m, "a", src, sources), None
        elif name in ("EXT", "INS"):
            a, b = _reg_operand(m, "a", rs, sources), Operand("b", (rd << 5) | sa, "pos/size fields")
        else:
            a, b = _reg_operand(m, "a", rs, sources), _reg_operand(m, "b", rt, sources)
        v.alu = AluView(_ALU_OPS[name], a, b, None, None)
    if name in _SIZES or (fam in ("load", "store")):
        if name.endswith("XC1"):
            off = m.read_reg(rt)
            base = _reg_operand(m, "base", rs, sources)
        elif word >> 26 == 0x1F:                       # EVA: 9-bit offset
            off = _sx((word >> 7) & 0x1FF, 9)
            base = _reg_operand(m, "base", rs, sources)
        else:
            off = _sx(imm, 16)
            base = _reg_operand(m, "base", rs, sources)
        addr = (base.value + off) & M32
        eva = word >> 26 == 0x1F
        size = _SIZES.get(name[:-1] if eva else name, 4)
        seg, priv, mapped = _segment(m, addr)
        access = "store" if (fam == "store" or name.startswith("S")) else "load"
        unaligned_ok = name[:3] in ("LWL", "LWR", "SWL", "SWR") or name in ("LUXC1", "SUXC1")
        v.address = AddressView(base, off, addr, size, access, seg, priv,
                                mapped, False, unaligned_ok or addr % size == 0)
        v.alu = AluView("a + b (effective address)", base,
                        Operand("b", off & M32, "offset (sign-extended)"), addr, addr == 0)
    v._before = _snapshot(m)
    m._mem_log = []
    if m.caches is not None:
        v.fetch = list(m.fetch_accesses)
        if m.caches.log is None:
            m.caches.watch()
    if hasattr(m.mmu, "watch"):
        v.fetch_tlb = list(m.fetch_tlb)
        if m.mmu.log is None:
            m.mmu.watch()
    return v


def finish(m, v: InstrView, *, result: int | None = None,
           exception: Any = None) -> InstrView:
    """Complete the view after the executor ran (or raised)."""
    log, m._mem_log = m._mem_log or [], None
    if m.caches is not None and m.caches.log is not None:
        v.cache = m.caches.taken()
    if hasattr(m.mmu, "watch") and m.mmu.log is not None:
        v.tlb = m.mmu.taken()
    if exception is not None:
        v.exception = C.EXC_NAMES.get(getattr(exception, "code", -1), type(exception).__name__)
        if v.alu is not None and getattr(exception, "code", None) == C.OV:
            v.alu.overflow = True
    v.reg_writes.extend(_changes(m, v._before))
    for entry in log:
        xlat = entry[-1]
        if entry[0] == "read":
            v.mem_reads.append(entry[1:-1])
        else:
            v.mem_writes.append(entry[1:-1])
        if v.address is not None and v.address.physical is None:
            v.address.physical, v.address.cca = xlat.pa, xlat.cca
            v.address.cached, v.address.how = xlat.cacheable, xlat.how
    if v.alu is not None and v.alu.result is None and exception is None:
        from .pipeline import analyze
        isa = _twin_isa(m) if v.compressed else m.isa
        dests = analyze(isa.decode(v.word)).writes
        gpr = [r for r in dests if isinstance(r, tuple) and r[0] == "r"]
        if gpr:
            v.alu.result = m.read_reg(gpr[0][1])
        elif "lo" in dests:
            v.alu.result = m.lo
        if v.alu.result is not None:
            v.alu.zero = v.alu.result == 0
    _branch_view(m, v, result)
    b = v.branch
    if b is not None and b.target is not None and m.compressed is not None:
        b.target &= ~1                  # (bit 0: the ISA mode there, not an address bit)
    v._before = ()
    return v


def _twin_isa(m):
    """The table the compressed instruction's MIPS32 twin decodes with."""
    if m.compressed == "micromips" and m._mm is not None:
        return m._twin_isa(m._mm)
    return ISA.for_release("R5")                    # (MIPS16e SEB on R1 too)


def _branch_view(m, v: InstrView, result: int | None) -> None:
    from .sim import NULLIFY
    if v.format == "MIPS16e":
        from .mips16 import ENTRIES
        e = ENTRIES[v.name[4:]]
    elif v.format == "microMIPS":
        from .micromips import ENTRIES
        e = ENTRIES[v.name[3:]]
    else:
        e = m.isa.get(v.name)
    if e["family"] not in ("branch", "branch_fpu", "jump", "branch_compact",
                           "jump_compact") or v.exception:
        return
    if e["family"] in ("branch_compact", "jump_compact"):
        target = m._redirect
        link = next((new for k, old, new in v.reg_writes if k == "$ra"), None)
        v.branch = BranchView("compact (no delay slot)", target is not None,
                              target, link=link)
        return
    w = v.word
    link = next((new for k, old, new in v.reg_writes if k == "$ra"), None)
    if e["family"] == "jump":
        v.branch = BranchView("always (jump)", True, result, link=link)
        return
    if v.format == "microMIPS":
        f, form = m._mm.f, m._mm.form
        r = lambda k: gpr_name(f[k])            # noqa: E731
        text = {"BEQ": lambda: f"{r('s')} == {r('t')}", "BNE": lambda: f"{r('s')} != {r('t')}",
                "BEQZ16": lambda: f"{r('s')} == 0", "BNEZ16": lambda: f"{r('s')} != 0",
                "BGEZ": lambda: f"{r('s')} >= 0", "BGEZAL": lambda: f"{r('s')} >= 0",
                "BGEZALS": lambda: f"{r('s')} >= 0", "BLTZ": lambda: f"{r('s')} < 0",
                "BLTZAL": lambda: f"{r('s')} < 0", "BLTZALS": lambda: f"{r('s')} < 0",
                "BGTZ": lambda: f"{r('s')} > 0", "BLEZ": lambda: f"{r('s')} <= 0",
                "B16": lambda: "always"}.get(form, lambda: "FPU condition code")()
        v.branch = BranchView(text, result is not None, result, link=link)
        return
    rs, rt = (w >> 21) & 31, (w >> 16) & 31
    name = v.name
    cond = {
        "BEQ": f"{gpr_name(rs)} == {gpr_name(rt)}", "BNE": f"{gpr_name(rs)} != {gpr_name(rt)}",
        "BGEZ": f"{gpr_name(rs)} >= 0", "BLTZ": f"{gpr_name(rs)} < 0",
        "BGTZ": f"{gpr_name(rs)} > 0", "BLEZ": f"{gpr_name(rs)} <= 0",
    }
    base = name.rstrip("L") if name not in ("BAL", "NAL", "B") else name
    base = base.replace("AL", "") if base.startswith(("BGEZAL", "BLTZAL")) else base
    text = cond.get(base, "FPU condition code" if name.startswith("BC") else
                    "always" if name in ("B", "BAL") else "never" if name == "NAL" else name)
    taken = result is not None and result != NULLIFY
    target = result if taken else None
    v.branch = BranchView(text, taken, target,
                          likely=name.endswith("L") and name not in ("BAL", "NAL"),
                          link=link)


def _snapshot(m) -> tuple:
    """The architectural state an instruction can change, as raw values
    (cheap: compared after the instruction, named only where it changed)."""
    cp0 = m.cp0
    # the set the instruction runs in (an ERET may switch to another: its
    # registers are not "written") and, with shadow sets, the previous set
    # (WRPGPR writes there)
    pss = m.srs_field(6) if m.hss else 0
    prev = (m.shadow[pss], list(m.shadow[pss])) if m.shadow[pss] is not m.regs else None
    return ((m.regs, list(m.regs)), m.hi, m.lo, list(m.fpr), m.fcsr,
            tuple(cp0), [cell["value"] for cell in cp0.values()], (pss, prev))


def _changes(m, before: tuple) -> list[tuple[str, int, int]]:
    """``(name, old, new)`` of everything that differs from ``before``,
    sorted by name; Count (CP0 9.0) ticks by itself and is left out."""
    (cur, regs), hi, lo, fpr, fcsr, cp0_keys, cp0_vals, (pss, prev) = before
    out = []
    for i in range(1, 32):
        if (regs[i] & M32) != (cur[i] & M32):
            out.append((gpr_name(i), regs[i] & M32, cur[i] & M32))
    if prev is not None:
        bank, old = prev
        for i in range(1, 32):
            if (old[i] & M32) != (bank[i] & M32):
                out.append((f"{gpr_name(i)} (set {pss})", old[i] & M32, bank[i] & M32))
    if hi != m.hi:
        out.append(("hi", hi, m.hi))
    if lo != m.lo:
        out.append(("lo", lo, m.lo))
    for i, (old, new) in enumerate(zip(fpr, m.fpr)):
        if old != new:
            out.append((f"$f{i}", old, new))
    if fcsr != m.fcsr:
        out.append(("fcsr", fcsr, m.fcsr))
    for (r, s), old in zip(cp0_keys, cp0_vals):
        new = m.cp0[(r, s)]["value"]
        if old != new and (r, s) != (9, 0):
            out.append((f"cp0:{r}.{s}", old, new))
    return sorted(out, key=lambda x: x[0])


# --------------------------------------------------------------- printing

def format_view(v: InstrView) -> str:
    """Multi-line, human-readable rendering of a view."""
    lines = [f"0x{v.pc:08x}: {v.text}" + ("   [delay slot]" if v.in_delay_slot else "")
             + f"   ({v.format}-format, 0x{v.word:08x})"]
    if v.compressed:
        lines.append(f"  {v.compressed}")
    lines.append("  decode: " + " | ".join(
        f"{f.name}[{f.msb}:{f.lsb}]={f.bits} {f.meaning}" if f.msb != f.lsb
        else f"{f.name}[{f.msb}]={f.bits} {f.meaning}" for f in v.fields))
    if v.alu:
        a = f"a = 0x{v.alu.a.value:08x} from {v.alu.a.source}" if v.alu.a else ""
        b = f"b = 0x{v.alu.b.value:08x} from {v.alu.b.source}" if v.alu.b else ""
        res = (f" -> 0x{v.alu.result:08x}" if v.alu.result is not None else "")
        flags = " zero" if v.alu.zero else ""
        flags += " OVERFLOW" if v.alu.overflow else ""
        lines.append(f"  ALU: {v.alu.op}; " + "; ".join(x for x in (a, b) if x) + res + flags)
    if v.address:
        ad = v.address
        lines.append(f"  address: base 0x{ad.base.value:08x} + {ad.offset} = 0x{ad.address:08x}"
                     f" ({ad.access} {ad.size} byte(s); {ad.segment}, {ad.privilege}"
                     f"{'' if ad.aligned else ', MISALIGNED'})")
        if ad.physical is not None:
            lines.append(f"  physical: 0x{ad.physical:08x} ({ad.how}; "
                         f"attribute {ad.cca} = {cca_name(ad.cca)})")
    if v.branch:
        br = v.branch
        what = f"taken -> 0x{br.target:08x}" if br.taken else (
            "not taken, delay slot nullified" if br.likely else "not taken")
        lines.append(f"  branch: {br.condition}: {what}"
                     + (f"; link $ra = 0x{br.link:08x}" if br.link is not None else ""))
    for addr, size, value in v.mem_reads:
        lines.append(f"  memory read  0x{addr:08x} ({size} byte(s)) = 0x{value:0{2 * size}x}")
    for addr, size, old, new in v.mem_writes:
        lines.append(f"  memory write 0x{addr:08x} ({size} byte(s)): "
                     f"0x{old:0{2 * size}x} -> 0x{new:0{2 * size}x}")
    for acc in v.fetch_tlb:
        lines.append(f"  TLB: {acc.describe()}")
    for acc in v.fetch:
        lines.append(f"  fetch: {acc.describe()}")
    for acc in v.tlb:
        lines.append(f"  TLB: {acc.describe()}")
    for acc in v.cache:
        lines.append(f"  cache: {acc.describe()}")
    for name, old, new in v.reg_writes:
        lines.append(f"  {name}: 0x{old:08x} -> 0x{new:08x}")
    if v.exception:
        lines.append(f"  EXCEPTION: {v.exception}")
    return "\n".join(lines)
