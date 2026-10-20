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

from dataclasses import dataclass
from operator import itemgetter
from typing import Any

from . import cp0 as C
from .disasm import text_at
from .isa import ISA, Decode
from .mmu import cca_name, segment
from .pipeline import analyze
from .sim import NULLIFY
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


class InstrView:
    """What happened inside one executed instruction.

    :func:`begin` / :func:`finish` only record the raw facts (the state
    before and after, the memory and cache / TLB logs, the outcome); the
    parts below marked *built* are made from them the first time one of
    them is read (PLAN M20: most views are never shown — a run traces
    every instruction, the views show the last one)."""

    #: built when first read (see :meth:`_build`)
    _BUILT = frozenset(("text", "format", "fields", "alu", "address", "branch", "reg_writes",
                        "mem_reads", "mem_writes", "compressed", "source_text"))

    pc: int
    word: int
    name: str
    in_delay_slot: bool
    exception: str | None
    #: the cache accesses of the instruction's fetch and of its own
    #: memory accesses (emips.cache.CacheAccess; machines with caches)
    fetch: list
    cache: list
    #: the translations (emips.tlb.TLBAccess) of the fetch and of the
    #: instruction (its accesses, TLBWI / TLBWR / TLBP / TLBR ...): a TLB
    #: or BAT MMU
    fetch_tlb: list
    tlb: list
    # built:
    text: str
    format: str
    fields: list[Field]
    alu: AluView | None
    address: AddressView | None
    branch: BranchView | None
    reg_writes: list[tuple[str, int, int]]
    mem_reads: list[tuple[int, int, int]]
    mem_writes: list[tuple[int, int, int, int]]
    #: MIPS16e / microMIPS: the compressed instruction (what the fields of
    #: a recoded one show is its MIPS32 twin) ...
    compressed: str
    #: ... its text alone (what a listing shows at that address)
    source_text: str

    def __init__(self, pc: int, word: int, name: str, in_slot: bool, raw: tuple) -> None:
        self.pc, self.word, self.name, self.in_delay_slot = pc, word, name, in_slot
        self.exception = None
        self.fetch, self.cache, self.fetch_tlb, self.tlb = [], [], [], []
        self._raw = raw              # (dec, sources, _Frozen, before) + finish's

    def __getattr__(self, name: str):
        if name in InstrView._BUILT and "_raw" in self.__dict__:
            _build(self)
            return self.__dict__[name]
        raise AttributeError(name)

    def __repr__(self) -> str:
        return f"InstrView(0x{self.pc:08x} {self.name})"


class _Frozen:
    """The machine as the view's builders read it — as the instruction
    found it (registers, modes, the compressed instruction) and left it
    (``post``: the registers, LO, the redirect): by the time a view is
    built the machine has moved on."""

    __slots__ = ("regs", "supervisor", "isa", "twin", "compressed", "isa_mode", "mm", "m16",
                 "release", "post", "lo", "redirect")

    def read_reg(self, n: int) -> int:
        n &= 31
        return self.regs[n] & M32 if n else 0

    def read_post(self, n: int) -> int:
        n &= 31
        return self.post[n] & M32 if n else 0


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
            if e.get("ase") in ("dsp", "dspr2") and w >> 26 == 0x1F and name == "sa":
                name = "op"                      # the DSP Module's sub-opcode (MD00374 5.1)
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
    if kind in ("pos_ext", "pos_ins") or kind.startswith("pos_d"):
        return "lsb" if lsb == 6 else ("msbd" if kind.startswith(("pos_ext", "pos_dext"))
                                       else "msb")
    return {"br": "offset", "jt": "instr_index", "uimm": "immediate",
            "imm": "immediate", "ac21": "ac", "dsa3": "sa", "dsa4": "sa", "dsa5": "sa",
            "dsa": "sa", "dsize": "size", "shilo": "shift", "rmask": "mask", "wmask": "mask",
            "dimm10": "immediate", "dimm8": "immediate", "bp2": "bp"}.get(kind, kind)


#: SPECIAL3 function -> the DSP Module's instruction class (MD00374 Table 5.3)
_DSP_CLASSES = {0x0A: "LX", 0x0C: "INSV", 0x10: "ADDU.QB", 0x11: "CMPU.EQ.QB",
                0x12: "ABSQ_S.PH", 0x13: "SHLL.QB", 0x18: "ADDUH.QB", 0x30: "DPA.W.PH",
                0x31: "APPEND", 0x38: "EXTR.W"}


def _sx(v: int, bits: int) -> int:
    return v - (1 << bits) if v >> (bits - 1) & 1 else v


def _meaning(name: str, v: int, width: int, dec: Decode) -> str:
    if name in ("rs", "rt", "rd", "base", "index") and width == 5:
        return gpr_name(v)
    if name in ("fs", "ft", "fd", "fr"):
        return f"$f{v}"
    if name == "op" and width == 5:          # the DSP Module's op field
        return dec.entry.get("mnemonic", dec.name)
    if name == "op":
        return _OPCODE_NAMES.get(v, dec.name)
    if (name == "funct" and dec.entry.get("ase") in ("dsp", "dspr2")
            and dec.word >> 26 == 0x1F):
        return f"the {_DSP_CLASSES.get(v, '?')} class (MD00374 Table 5.3)"
    if name == "funct":
        return dec.name
    if name == "ac":
        return f"$ac{v}"
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


def _reg_operand(st: _Frozen, name: str, n: int, sources: dict | None) -> Operand:
    src = (sources or {}).get(("r", n), "register file")
    return Operand(name, st.read_reg(n), f"{gpr_name(n)} ({src})")


def _segment(st: _Frozen, addr: int) -> tuple[str, str, bool]:
    """(segment, privilege, mapped) of a virtual address."""
    seg = segment(addr)
    if seg == "kuseg":
        return seg, "user", True
    if seg == "kseg2" and st.supervisor:
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


def _compressed(st: _Frozen, pc: int) -> tuple[str, str, int, int] | None:
    """(the ISA's name, the text, the encoding, its size) of the MIPS16e /
    microMIPS instruction the machine ran, or None in MIPS32 mode."""
    if not st.isa_mode:
        return None
    if st.compressed == "micromips":
        i = st.mm
        if i is None:
            return None
        from . import micromips
        return "microMIPS", micromips.text(i, pc, release=st.release), i.raw, i.size
    i = st.m16
    if i is None:
        return None
    from . import mips16
    return "MIPS16e", mips16.text(i, pc), i.raw, i.size


def begin(m, pc: int, word: int, dec: Decode, in_slot: bool = False,
          sources: dict | None = None) -> InstrView:
    """Record the state before the instruction (``sources``: where its
    register operands come from, the pipeline's bypasses); call
    :func:`finish` afterwards."""
    st = _Frozen()
    st.isa_mode = getattr(m, "isa_mode", 0)
    st.compressed, st.mm, st.m16 = m.compressed, m._mm, getattr(m, "_m16", None)
    st.isa, st.release, st.supervisor = m.isa, m.release, m.mmu.supervisor
    st.twin = _twin_isa(m) if st.isa_mode else None
    special = dec.entry.get("isa") in ("mips16e", "micromips")   # no MIPS32 twin
    before = _snapshot(m, special or dec.entry["family"] not in _NO_CP0)
    st.regs = before[0][1]
    v = InstrView(pc, word, dec.name, in_slot, (dec, sources, st, before))
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
    """Record what the executor did (or that it raised ``exception``)."""
    log, m._mem_log = m._mem_log or [], None
    if m.caches is not None and m.caches.log is not None:
        v.cache = m.caches.taken()
    if hasattr(m.mmu, "watch") and m.mmu.log is not None:
        v.tlb = m.mmu.taken()
    code = None
    if exception is not None:
        code = getattr(exception, "code", None)
        v.exception = C.EXC_NAMES.get(code if code is not None else -1,
                                      type(exception).__name__)
    dec, sources, st, before = v._raw
    after = _after(m, before)
    st.post, st.lo, st.redirect = after[0], m.lo, m._redirect
    v._raw = (dec, sources, st, before, after, log, result, exception is not None, code)
    return v


def _build(v: InstrView) -> None:
    """Make the view's text, fields, ALU / address / branch parts and
    effects from what :func:`begin` / :func:`finish` recorded."""
    raw = v.__dict__.pop("_raw")
    if len(raw) == 4:                   # (read before finish: what is known)
        raw = raw + (None, [], None, False, None)
    dec, sources, st, before, after, log, result, failed, code = raw
    pc, word, name, fam = v.pc, v.word, v.name, dec.entry["family"]
    d = v.__dict__
    d.update(alu=None, address=None, branch=None, compressed="", source_text="",
             mem_reads=[], mem_writes=[])
    c = _compressed(st, pc)
    if dec.entry.get("isa") in ("mips16e", "micromips"):    # no MIPS32 twin: its own view
        label, text, raw16, size = c
        d.update(text=text, format=label, fields=[], source_text=text,
                 compressed=f"{label} {text} (0x{raw16:0{2 * size}x})")
    else:
        isa = st.isa if c is None else st.twin
        d.update(text=text_at(word, pc, isa), format=instruction_format(word),
                 fields=decode_fields(dec))
        if c is not None:
            label, text, raw16, size = c
            v.compressed = (f"{label} {text} (0x{raw16:0{2 * size}x}), "
                            "recoded as this MIPS32 instruction")
            v.source_text = text
        _operands(v, st, word, name, fam, sources)
    v.reg_writes = _diff(before, after) if after is not None else []
    if code == C.OV and v.alu is not None:
        v.alu.overflow = True
    for entry in log:
        xlat = entry[-1]
        if entry[0] == "read":
            v.mem_reads.append(entry[1:-1])
        else:
            v.mem_writes.append(entry[1:-1])
        if v.address is not None and v.address.physical is None:
            v.address.physical, v.address.cca = xlat.pa, xlat.cca
            v.address.cached, v.address.how = xlat.cacheable, xlat.how
    if after is None:
        return
    if v.alu is not None and v.alu.result is None and not failed:
        isa = st.twin if v.compressed else st.isa
        dests = analyze(isa.decode(word)).writes
        gpr = [r for r in dests if isinstance(r, tuple) and r[0] == "r"]
        if gpr:
            v.alu.result = st.read_post(gpr[0][1])
        elif "lo" in dests:
            v.alu.result = st.lo
        if v.alu.result is not None:
            v.alu.zero = v.alu.result == 0
    _branch_view(st, v, result)
    b = v.branch
    if b is not None and b.target is not None and st.compressed is not None:
        b.target &= ~1                  # (bit 0: the ISA mode there, not an address bit)


def _operands(v: InstrView, st: _Frozen, word: int, name: str, fam: str,
              sources: dict | None) -> None:
    """The ALU's operands and the address of a load / store, as the
    instruction read them."""
    rs, rt, rd, sa = (word >> 21) & 31, (word >> 16) & 31, (word >> 11) & 31, (word >> 6) & 31
    imm = word & 0xFFFF
    if name in _ALU_OPS:
        if name in _SHIFT_IMM:
            a, b = _reg_operand(st, "a", rt, sources), Operand("b", sa, "shift amount (sa field)")
        elif name in _SHIFT_VAR:
            a, b = _reg_operand(st, "a", rt, sources), _reg_operand(st, "b", rs, sources)
        elif fam == "imm":
            val = imm if name in _ZEXT_IMM else _sx(imm, 16) & M32
            how = "zero-extended" if name in _ZEXT_IMM else "sign-extended"
            a = None if name == "LUI" else _reg_operand(st, "a", rs, sources)
            b = Operand("b", val, f"immediate 0x{imm:04x} ({how})")
        elif name in ("CLZ", "CLO", "SEB", "SEH", "WSBH"):
            src = rs if name in ("CLZ", "CLO") else rt
            a, b = _reg_operand(st, "a", src, sources), None
        elif name in ("EXT", "INS"):
            a, b = _reg_operand(st, "a", rs, sources), Operand("b", (rd << 5) | sa, "pos/size fields")
        else:
            a, b = _reg_operand(st, "a", rs, sources), _reg_operand(st, "b", rt, sources)
        v.alu = AluView(_ALU_OPS[name], a, b, None, None)
    if name in _SIZES or (fam in ("load", "store")):
        if name.endswith("XC1"):
            off = st.read_reg(rt)
            base = _reg_operand(st, "base", rs, sources)
        elif word >> 26 == 0x1F:                       # EVA: 9-bit offset
            off = _sx((word >> 7) & 0x1FF, 9)
            base = _reg_operand(st, "base", rs, sources)
        else:
            off = _sx(imm, 16)
            base = _reg_operand(st, "base", rs, sources)
        addr = (base.value + off) & M32
        eva = word >> 26 == 0x1F
        size = _SIZES.get(name[:-1] if eva else name, 4)
        seg, priv, mapped = _segment(st, addr)
        access = "store" if (fam == "store" or name.startswith("S")) else "load"
        unaligned_ok = name[:3] in ("LWL", "LWR", "SWL", "SWR") or name in ("LUXC1", "SUXC1")
        v.address = AddressView(base, off, addr, size, access, seg, priv,
                                mapped, False, unaligned_ok or addr % size == 0)
        v.alu = AluView("a + b (effective address)", base,
                        Operand("b", off & M32, "offset (sign-extended)"), addr, addr == 0)


def _twin_isa(m):
    """The table the compressed instruction's MIPS32 twin decodes with."""
    if m.compressed == "micromips" and m._mm is not None:
        return m._twin_isa(m._mm)
    return ISA.for_release("R5")                    # (MIPS16e SEB on R1 too)


def _branch_view(st: _Frozen, v: InstrView, result: int | None) -> None:
    if v.format == "MIPS16e":
        from .mips16 import ENTRIES
        e = ENTRIES[v.name[4:]]
    elif v.format == "microMIPS":
        from .micromips import ENTRIES
        e = ENTRIES[v.name[3:]]
    else:
        e = st.isa.get(v.name)
    if e["family"] not in ("branch", "branch_fpu", "jump", "branch_compact",
                           "jump_compact") or v.exception:
        return
    if e["family"] in ("branch_compact", "jump_compact"):
        target = st.redirect
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
        f, form = st.mm.f, st.mm.form
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


_VALUE = itemgetter("value")


#: families whose executors change no CP0 register (an exception they
#: raise is taken after the view is finished): their views skip comparing
#: the CP0 cells, the larger part of a view's cost
_NO_CP0 = frozenset(("alu", "imm", "shift", "branch", "branch_compact", "jump",
                     "jump_compact", "muldiv"))


def _snapshot(m, cp0: bool = True) -> tuple:
    """The architectural state an instruction can change, as raw values
    (cheap: compared after the instruction, named only where it changed;
    without ``cp0``, the CP0 registers are left out)."""
    # the set the instruction runs in (an ERET may switch to another: its
    # registers are not "written") and, with shadow sets, the previous set
    # (WRPGPR writes there)
    pss = m.srs_field(6) if m.hss else 0
    prev = (m.shadow[pss], m.shadow[pss][:]) if m.shadow[pss] is not m.regs else None
    cells = m.cp0
    keys, vals = (tuple(cells), list(map(_VALUE, cells.values()))) if cp0 else (None, None)
    return ((m.regs, m.regs[:]), m.hi, m.lo, m.fpr[:], m.fcsr, keys, vals, (pss, prev),
            (m.dspctl, m.dsp_hi[:], m.dsp_lo[:]))


def _gpr_changes(old: list, cur: list, suffix: str = "") -> list:
    if old == cur:
        return []
    return [(gpr_name(i) + suffix, a & M32, b & M32)
            for i, (a, b) in enumerate(zip(old, cur)) if i and a != b and (a ^ b) & M32]


def _after(m, before: tuple) -> tuple:
    """The same state as :func:`_snapshot` recorded, as it is now (the
    register bank the instruction ran in, whatever an ERET switched to)."""
    (cur, _regs), _hi, _lo, _fpr, _fcsr, cp0_keys, _vals, (_pss, prev), _dsp = before
    vals = None
    if cp0_keys is not None:
        cp0 = m.cp0
        vals = list(map(_VALUE, cp0.values())) if tuple(cp0) == cp0_keys else \
            [cp0[k]["value"] for k in cp0_keys]     # (a cell added: by key)
    return (cur[:], m.hi, m.lo, m.fpr[:], m.fcsr, vals,
            prev[0][:] if prev is not None else None, (m.dspctl, m.dsp_hi[:], m.dsp_lo[:]))


def _diff(before: tuple, after: tuple) -> list[tuple[str, int, int]]:
    """``(name, old, new)`` of everything that differs, sorted by name;
    Count (CP0 9.0) ticks by itself and is left out."""
    (_cur, regs), hi, lo, fpr, fcsr, cp0_keys, cp0_vals, (pss, prev), dsp = before
    cur, hi2, lo2, fpr2, fcsr2, now, bank, (dspctl2, dhi2, dlo2) = after
    out = _gpr_changes(regs, cur)
    if prev is not None:
        out += _gpr_changes(prev[1], bank, f" (set {pss})")
    if hi != hi2:
        out.append(("hi", hi, hi2))
    if lo != lo2:
        out.append(("lo", lo, lo2))
    if fpr != fpr2:
        for i, (old, new) in enumerate(zip(fpr, fpr2)):
            if old != new:
                out.append((f"$f{i}", old, new))
    if fcsr != fcsr2:
        out.append(("fcsr", fcsr, fcsr2))
    dspctl, dhi, dlo = dsp                    # the DSP Module: DSPControl, ac1-ac3
    if dspctl != dspctl2:
        out.append(("dspctl", dspctl, dspctl2))
    if dhi != dhi2 or dlo != dlo2:
        for n in range(1, 4):
            if dhi[n] != dhi2[n]:
                out.append((f"hi{n}", dhi[n], dhi2[n]))
            if dlo[n] != dlo2[n]:
                out.append((f"lo{n}", dlo[n], dlo2[n]))
    if cp0_keys is not None and now != cp0_vals:
        for (r, s), old, new in zip(cp0_keys, cp0_vals, now):
            if old != new and (r, s) != (9, 0):
                out.append((f"cp0:{r}.{s}", old, new))
    return sorted(out, key=_NAME) if len(out) > 1 else out


def _changes(m, before: tuple) -> list[tuple[str, int, int]]:
    """:func:`_diff` of ``before`` and the machine now."""
    return _diff(before, _after(m, before))


_NAME = itemgetter(0)


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
