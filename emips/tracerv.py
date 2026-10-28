"""What happens inside one RISC-V instruction (PLAN M25 phase 3c): the
:class:`emips.trace.InstrView` of a hart, so the debugger, ``run --trace``
and the window show a RISC-V instruction as they show a MIPS one.

:func:`begin` / :func:`finish` (called by ``RVMachine.step`` while
``tracing``) record the raw facts — the registers, FPRs, fcsr, the
privilege mode and the trap CSRs before and after, the memory log, the
next PC, the trap taken — and the view is built when first read:

* **decode** — every bit field, MSB first: the fixed bits named by the
  unprivileged spec's formats (``opcode`` with its major opcode name,
  ``funct3``, ``funct7`` ...; a compressed instruction's ``op`` quadrant,
  ``funct3`` / ``funct4`` / ``funct6``), the operand fields by
  riscv-opcodes' names and the immediate bits each holds (``imm[11:5]``);
* **ALU** — the operation, its inputs (``rs1``, ``rs2`` or the immediate,
  the PC) and the result;
* **address** — base + offset, the access, the privilege mode and the
  translation observed (Bare / Sv32 / Sv39 ...: the physical address);
* **branch** — the condition, taken or not, the target, the link;
* **effects** — x / f registers, fcsr, the CSRs, the privilege mode and
  memory as they changed, and the trap.

Observed, never recomputed: nothing here changes the hart.
"""

from __future__ import annotations

from .riscv import FPR_NAMES, GPR_NAMES, RM_NAMES, _ARGS, _fields, CSR_NAMES
from .trace import AddressView, AluView, BranchView, Field, InstrView, Operand

#: the trap / status CSRs a view compares (attributes of emips.rvcsr.CSRFile)
_TRACKED = (("mstatus", "mstatus"), ("mtvec", "mtvec"), ("mepc", "mepc"),
            ("mcause", "mcause"), ("mtval", "mtval"), ("mscratch", "mscratch"),
            ("mie", "mie"), ("stvec", "stvec"), ("sepc", "sepc"), ("scause", "scause"),
            ("stval", "stval"), ("sscratch", "sscratch"), ("satp", "satp"),
            # the hypervisor extension's (priv. ch. 21: trap state, the guest's CSRs)
            ("mtval2", "mtval2"), ("mtinst", "mtinst"), ("hstatus", "hstatus"),
            ("htval", "htval"), ("htinst", "htinst"), ("hgatp", "hgatp"),
            ("vsstatus", "vsstatus"), ("vstvec", "vstvec"), ("vsepc", "vsepc"),
            ("vscause", "vscause"), ("vstval", "vstval"), ("vsscratch", "vsscratch"),
            ("vsatp", "vsatp"),
            # Smrnmi's (priv. 8.2: an RNMI / MNRET changes them)
            ("mnstatus", "mnstatus"), ("mnepc", "mnepc"), ("mncause", "mncause"),
            # Zicfilp's landing-pad state, Zicfiss's ssp (LF10: unpriv. ch. 33)
            ("ELP", "elp"), ("ssp", "ssp"))
#: the base opcode map (unprivileged Table 70): bits 6:2 -> name
OPCODES = {0x00: "LOAD", 0x01: "LOAD-FP", 0x02: "custom-0", 0x03: "MISC-MEM",
           0x04: "OP-IMM", 0x05: "AUIPC",
           0x06: "OP-IMM-32", 0x08: "STORE", 0x09: "STORE-FP", 0x0B: "AMO", 0x0C: "OP",
           0x0D: "LUI", 0x0E: "OP-32", 0x10: "MADD", 0x11: "MSUB", 0x12: "NMSUB",
           0x13: "NMADD", 0x14: "OP-FP", 0x18: "BRANCH", 0x19: "JALR", 0x1B: "JAL",
           0x1C: "SYSTEM"}
#: a compressed instruction's format (unprivileged Table 33)
C_FORMATS = {
    **dict.fromkeys(("c.jr", "c.jalr", "c.mv", "c.add", "c.ebreak"), "CR"),
    **dict.fromkeys(("c.li", "c.lui", "c.addi", "c.addiw", "c.addi16sp", "c.slli",
                     "c.lwsp", "c.ldsp", "c.flwsp", "c.fldsp", "c.nop"), "CI"),
    **dict.fromkeys(("c.swsp", "c.sdsp", "c.fswsp", "c.fsdsp"), "CSS"),
    "c.addi4spn": "CIW",
    **dict.fromkeys(("c.lw", "c.ld", "c.flw", "c.fld"), "CL"),
    **dict.fromkeys(("c.sw", "c.sd", "c.fsw", "c.fsd"), "CS"),
    **dict.fromkeys(("c.sub", "c.xor", "c.or", "c.and", "c.subw", "c.addw"), "CA"),
    **dict.fromkeys(("c.beqz", "c.bnez", "c.srli", "c.srai", "c.andi"), "CB"),
    **dict.fromkeys(("c.j", "c.jal"), "CJ"),
}
#: the ALU operation of an instruction (its 32-bit row's name)
_ALU = {
    "add": "a + b", "addi": "a + b", "sub": "a - b", "and": "a & b", "andi": "a & b",
    "or": "a | b", "ori": "a | b", "xor": "a ^ b", "xori": "a ^ b",
    "sll": "a << b", "slli": "a << b", "srl": "a >> b (logical)",
    "srli": "a >> b (logical)", "sra": "a >> b (arithmetic)", "srai": "a >> b (arithmetic)",
    "slt": "a < b (signed) ? 1 : 0", "slti": "a < b (signed) ? 1 : 0",
    "sltu": "a < b (unsigned) ? 1 : 0", "sltiu": "a < b (unsigned) ? 1 : 0",
    "lui": "b (the upper immediate)", "auipc": "pc + b",
    "addw": "a + b (32-bit, sign-extended)", "addiw": "a + b (32-bit, sign-extended)",
    "subw": "a - b (32-bit, sign-extended)", "sllw": "a << b (32-bit)",
    "slliw": "a << b (32-bit)", "srlw": "a >> b (32-bit logical)",
    "srliw": "a >> b (32-bit logical)", "sraw": "a >> b (32-bit arithmetic)",
    "sraiw": "a >> b (32-bit arithmetic)",
    "mul": "a × b (low)", "mulh": "a × b (high, signed)", "mulhsu": "a × b (high, s × u)",
    "mulhu": "a × b (high, unsigned)", "div": "a / b (signed)", "divu": "a / b (unsigned)",
    "rem": "a % b (signed)", "remu": "a % b (unsigned)", "mulw": "a × b (32-bit)",
    "divw": "a / b (32-bit signed)", "divuw": "a / b (32-bit unsigned)",
    "remw": "a % b (32-bit signed)", "remuw": "a % b (32-bit unsigned)",
    "sh1add": "(a << 1) + b", "sh2add": "(a << 2) + b", "sh3add": "(a << 3) + b",
    "add.uw": "zext32(a) + b", "sh1add.uw": "(zext32(a) << 1) + b",
    "sh2add.uw": "(zext32(a) << 2) + b", "sh3add.uw": "(zext32(a) << 3) + b",
    "slli.uw": "zext32(a) << b",
    "andn": "a & ~b", "orn": "a | ~b", "xnor": "~(a ^ b)", "clz": "leading zeros of a",
    "ctz": "trailing zeros of a", "cpop": "ones in a", "clzw": "leading zeros of a[31:0]",
    "ctzw": "trailing zeros of a[31:0]", "cpopw": "ones in a[31:0]",
    "max": "max(a, b) (signed)", "maxu": "max(a, b) (unsigned)",
    "min": "min(a, b) (signed)", "minu": "min(a, b) (unsigned)",
    "sext.b": "sign-extend a[7:0]", "sext.h": "sign-extend a[15:0]",
    "zext.h": "zero-extend a[15:0]", "rol": "a rotated left by b", "ror": "a rotated right by b",
    "rori": "a rotated right by b", "rolw": "a[31:0] rotated left by b",
    "rorw": "a[31:0] rotated right by b", "roriw": "a[31:0] rotated right by b",
    "orc.b": "each byte of a: 0x00 or 0xff", "rev8": "a's bytes reversed",
    # LF1: Zbs, Zbc / Zbkc, Zbkb, Zbkx, Zicond (the MOPs read nothing: no ALU view)
    "bclr": "a with bit b cleared", "bclri": "a with bit b cleared",
    "bset": "a with bit b set", "bseti": "a with bit b set",
    "binv": "a with bit b inverted", "binvi": "a with bit b inverted",
    "bext": "bit b of a", "bexti": "bit b of a",
    "clmul": "a ⊗ b (carry-less, low)", "clmulh": "a ⊗ b (carry-less, high)",
    "clmulr": "a ⊗ b (carry-less, reversed: bits 2·XLEN-2 .. XLEN-1)",
    "pack": "b[low half] : a[low half]", "packh": "zext(b[7:0] : a[7:0])",
    "packw": "b[15:0] : a[15:0] (sign-extended)", "brev8": "each byte of a bit-reversed",
    "zip": "a's halves interleaved (low -> even bits)",
    "unzip": "a's even bits -> low half, odd bits -> high half",
    "xperm4": "each nibble of b picks a nibble of a (0 when out of range)",
    "xperm8": "each byte of b picks a byte of a (0 when out of range)",
    "czero.eqz": "b == 0 ? 0 : a", "czero.nez": "b != 0 ? 0 : a",
}
_BRANCH = {"beq": ("==", False), "bne": ("!=", False), "blt": ("<", True),
           "bge": (">=", True), "bltu": ("<", False), "bgeu": (">=", False)}
_LOADS = {"lb": 1, "lh": 2, "lw": 4, "ld": 8, "lbu": 1, "lhu": 2, "lwu": 4, "flw": 4,
          "fld": 8, "flh": 2, "flq": 16, "mips.ldp": 16, "mips.lwp": 8}
_STORES = {"sb": 1, "sh": 2, "sw": 4, "sd": 8, "fsw": 4, "fsd": 8, "fsh": 2, "fsq": 16,
           "mips.sdp": 16, "mips.swp": 8}         # (the P8700's pairs: both halves)
#: the hypervisor's loads / stores (priv. 21.3.1: as VS / VU-mode, by hstatus.SPVP)
_HLOADS = {"hlv.b": 1, "hlv.bu": 1, "hlv.h": 2, "hlv.hu": 2, "hlv.w": 4, "hlv.wu": 4,
           "hlv.d": 8, "hlvx.hu": 2, "hlvx.wu": 4}
_HSTORES = {"hsv.b": 1, "hsv.h": 2, "hsv.w": 4, "hsv.d": 8}
_LOADS.update(_HLOADS)
_STORES.update(_HSTORES)
_HSTATUS = next(k for k, (n, _a) in enumerate(_TRACKED) if n == "hstatus")


def mode_name(priv: int, v: int, h: bool = False) -> str:
    """A privilege mode's name (priv. 21.1 Table 38): M, S (with H: HS), U,
    VS, VU."""
    if v:
        return "VS" if priv == 1 else "VU"
    return {3: "M", 1: "HS" if h else "S", 0: "U"}[priv]


class RVView(InstrView):
    """The view of one RISC-V instruction (see the module docstring)."""

    rv = True

    def __getattr__(self, name: str):
        if name in InstrView._BUILT and "_raw" in self.__dict__:
            _build_rv(self)
            return self.__dict__[name]
        raise AttributeError(name)


def _vector_state(m):
    """The vector unit as a view compares it (LF6): v0-v31 and the CSRs."""
    vu = m.vector
    if vu is None:
        return None
    return (bytes(vu.regs), vu.vl, vu.vtype, vu.vstart, vu.vxsat, vu.vxrm, vu.sew, vu.vlenb)


def begin(m, pc: int) -> RVView:
    c = m.csrs
    before = (list(m.regs), list(m.fpr), m.fcsr, (m.priv, m.v),
              tuple(getattr(c, attr) for _n, attr in _TRACKED), _vector_state(m))
    v = RVView(pc, 0, "", False, (m, pc, before))
    v.digits = 8 if m.xlen == 32 else 16
    v.vdigits = 2 * m.vector.vlenb if m.vector is not None else 32
    v.csr_before = None
    #: the operands a pipeline bypassed (emips.rvpipeline: ("r", n) -> "forwarded
    #: EX/MEM -> EX from ..."): the ALU / address views name the path
    v.sources = m.rv_sources or {}
    m._mem_log = []
    m.cmo_note = m.cmo_access = None
    if m.caches is not None:
        v.fetch = list(m.fetch_accesses)
        if m.caches.log is None:
            m.caches.watch()
    return v


def fetched(m, v: RVView, i) -> None:
    """The instruction is known (before it executes): a CSR instruction's
    CSR as it was."""
    if i.base.name.startswith("csrr") and i.f.get("csr") in m.csrs.table:
        try:
            v.csr_before = (i.f["csr"], m.csrs.read(i.f["csr"]))
        except Exception:                       # noqa: BLE001  (a view: never fails)
            pass


def finish(m, v: RVView, i, nxt: int | None, trap: str | None = None) -> RVView:
    """Record what the instruction ``i`` did (``i`` None: the fetch
    trapped); ``trap`` names the trap it took."""
    log, m._mem_log = m._mem_log or [], None
    if m.caches is not None and m.caches.log is not None:
        v.cache = m.caches.taken()
    _m, pc, before = v._raw
    c = m.csrs
    csr = None
    if v.csr_before is not None:
        try:
            csr = (*v.csr_before, c.read(v.csr_before[0]))
        except Exception:                       # noqa: BLE001
            csr = None
    after = (list(m.regs), list(m.fpr), m.fcsr, (m.priv, m.v),
             tuple(getattr(c, attr) for _n, attr in _TRACKED), _vector_state(m))
    if i is not None:
        v.word, v.size, v.name = i.raw, i.size, i.row.name
    else:
        v.name = "(fetch)"
    v.exception = trap
    if m.cmo_access is not None:                # (a CBO: its block, what it did)
        v._cmo = (m.cmo_access, m.cmo_note or "")
    v._raw = (m.xlen, m.flen, m.extensions, pc, before, after, log, i, nxt, csr, m.pc)
    return v


def _build_rv(v: RVView) -> None:
    xlen, flen, exts, pc, before, after, log, i, nxt, csr, now = v.__dict__.pop("_raw")
    mask = (1 << xlen) - 1
    v.fields, v.alu, v.address, v.branch, v.lanes = [], None, None, None, None
    v.compressed = v.source_text = ""
    if i is None:
        v.text, v.format = "(the fetch trapped)", "-"
    else:
        from .sessionrv import _disassembler
        v.text = _disassembler(xlen, exts, False).text(i.raw, i.size, pc)
        v.source_text = v.text
        v.format = _format(i)
        v.fields = fields(i)
        if i.size == 2:
            v.compressed = (f"compressed 16-bit {i.row.name} ({v.format}): the 32-bit "
                            f"{i.base.name}")
        _operands(v, i, pc, before, after, log, mask, nxt, "h" in exts)
    v.mem_reads = [(e[1], e[2], e[3]) for e in log if e[0] == "read"]
    v.mem_writes = [(e[1], e[2], e[3], e[4]) for e in log if e[0] == "write"]
    v.reg_writes = _effects(before, after, csr, flen)
    if after[5] is not None and before[5] is not None and before[5][0] != after[5][0]:
        v.lanes = VectorLanes(before[5], after[5])


#: OP-V's funct3 (30.5 / 30.10.1): the operand category
OPV = {0: "OPIVV", 1: "OPFVV", 2: "OPMVV", 3: "OPIVI", 4: "OPIVX", 5: "OPFVF", 6: "OPMVX",
       7: "OPCFG"}
#: the vector load / store addressing modes (30.7.2, mop)
MOP = {0: "unit-stride", 1: "indexed-unordered", 2: "strided", 3: "indexed-ordered"}


def _format(i) -> str:
    if i.size == 2:
        return C_FORMATS.get(i.row.name, "C")
    op = i.raw & 0x7F
    if i.row.file.startswith(("rv_v", "rv_zv")):
        if op == 0x57:
            return OPV[i.raw >> 12 & 7]
        if op in (0x07, 0x27):
            return f"V{'L' if op == 0x07 else 'S'} {MOP[i.raw >> 26 & 3]}"
        return "OP-VE"                          # (32.3: the vector crypto major opcode)
    args = set(i.base.args)
    if "rs3" in args:
        return "R4-type"
    if "jimm20" in args:
        return "J-type"
    if "imm20" in args:
        return "U-type"
    if "bimm12hi" in args:
        return "B-type"
    if "imm12hi" in args:
        return "S-type"
    if args & {"imm12", "shamtw", "shamtd", "csr", "zimm5"} or i.base.name in (
            "ecall", "ebreak", "mret", "sret", "wfi", "fence", "fence.i", "fence.tso",
            "pause", "dret"):
        return "I-type"
    return "R-type"


def _reg(n: int) -> str:
    return f"x{n} ({GPR_NAMES[n]})"


def _arg_meaning(name: str, a: str, value: int, xlen: int) -> str:
    if name in ("rd", "rs1", "rs2", "rs3"):
        return _reg(value)
    if name in ("vd", "vs1", "vs2", "vs3"):
        return f"v{value}"
    if name == "vm":
        return "unmasked" if value else "masked: v0.t"
    if name == "nf":
        return f"{value + 1} field(s) / register(s)"
    if name == "vtypei":
        from .riscv import vtype_text
        return f"vtype = {vtype_text(value)}"
    if name == "csr":
        return CSR_NAMES.get(value, f"CSR 0x{value:03x}")
    if name == "rm":
        return RM_NAMES[value] or "reserved"
    if name in ("pred", "succ"):
        return "".join(c for b, c in zip((8, 4, 2, 1), "iorw") if value & b) or "-"
    return f"{name} = {value}"


def fp_operands(name: str) -> frozenset:
    """The floating-point operands of an F / D / Q / Zfh / Zfa instruction
    (its 32-bit row's name; unprivileged ch. 20-25): emips.rvfpu's
    operand_formats."""
    from .rvfpu import operand_formats
    return frozenset(operand_formats(name))


def f_registers(i) -> frozenset:
    """The operands of a decoded instruction that are f registers (none
    with Zfinx & co.: ch. 26 puts them in the x registers)."""
    return frozenset() if i.inx else fp_operands(i.base.name)


def _reg_of(name: str, n: int, fp: frozenset) -> str:
    return f"f{n} ({FPR_NAMES[n]})" if name in fp else _reg(n)


def fields(i) -> list[Field]:
    """Every bit field of the instruction (its own row: a compressed one's
    16 bits), MSB first; an immediate split over several fields names the
    bits of the immediate each holds."""
    row, raw, n = i.row, i.raw, 8 * i.size
    own = _fields(row, raw)
    fp = f_registers(i)
    parts: list[tuple[int, int, str, str]] = []
    covered = 0
    last_of: dict[str, tuple[str, int]] = {}    # name -> (arg, segment) holding the total
    for a in row.args:
        names, segs, _w, _o = _ARGS[a]
        last_of[names[0]] = (a, len(segs) - 1)
    for a in row.args:
        names, segs, _width, offset = _ARGS[a]
        name = names[0]
        reg = name in ("rd", "rs1", "rs2", "rs3", "rd2")
        for k, (hi, lo, pos) in enumerate(segs):
            covered |= ((1 << (hi - lo + 1)) - 1) << lo
            last = last_of[name] == (a, k)
            if reg:
                label = name + ("'" if offset else "")
                meaning = _reg_of(name, own.get(name, 0), fp) if last else ""
            elif name in ("imm", "shamt") and (len(segs) > 1 or pos):
                top = pos + hi - lo
                label = name
                meaning = (f"{name} bit {pos}" if top == pos else f"{name} bits {top}:{pos}") \
                    + (f": {name} = {own.get(name, 0)}" if last else "")
            else:
                label = name
                meaning = _arg_meaning(name, a, own.get(name, 0), 0) if last else ""
            parts.append((hi, lo, label, meaning))
    vec = row.file.startswith(("rv_v", "rv_zv"))
    for hi, lo in _fixed_runs(~covered & ((1 << n) - 1), n):
        name = _vector_fixed(hi, lo, raw) if vec else None
        parts.append((hi, lo, name or _fixed_name(hi, lo, n), _fixed_meaning(hi, lo, n, raw)
                      if not name else _vector_meaning(hi, lo, raw)))
    out = []
    for hi, lo, label, meaning in sorted(parts, key=lambda p: -p[0]):
        bits = format((raw >> lo) & ((1 << (hi - lo + 1)) - 1), f"0{hi - lo + 1}b")
        out.append(Field(label, hi, lo, bits, int(bits, 2), meaning))
    return out


#: where a standard field starts (its MSB): a fixed run is split there
_BOUNDS = {32: (31, 24, 19, 14, 11, 6), 16: (1,)}


def _fixed_runs(bits: int, n: int) -> list[tuple[int, int]]:
    """The runs of fixed bits, MSB first, split where a field starts."""
    runs, cur = [], None
    for b in range(n - 1, -1, -1):
        on = bits >> b & 1
        if on and cur is not None and b in _BOUNDS[n]:
            runs.append((cur, b + 1))
            cur = None
        if on and cur is None:
            cur = b
        elif not on and cur is not None:
            runs.append((cur, b + 1))
            cur = None
    if cur is not None:
        runs.append((cur, 0))
    return runs


def _vector_fixed(hi: int, lo: int, raw: int) -> str | None:
    """The vector formats' fixed fields (30.5 / 30.7.1)."""
    op = raw & 0x7F
    if op in (0x07, 0x27):
        return {(28, 26): "mew mop", (27, 26): "mop", (28, 28): "mew", (14, 12): "width",
                (24, 20): "lumop" if op == 0x07 else "sumop"}.get((hi, lo))
    return {(31, 26): "funct6", (14, 12): "funct3", (19, 15): "vs1 (op)",
            (31, 25): "funct6 vm"}.get((hi, lo))


def _vector_meaning(hi: int, lo: int, raw: int) -> str:
    if (hi, lo) == (14, 12):
        if raw & 0x7F == 0x57:
            return OPV[raw >> 12 & 7]
        if raw & 0x7F in (0x07, 0x27):
            return {0: "EEW 8", 5: "EEW 16", 6: "EEW 32", 7: "EEW 64"}.get(raw >> 12 & 7, "")
    if (hi, lo) in ((28, 26), (27, 26)):
        return MOP[raw >> 26 & 3]
    return ""


def _fixed_name(hi: int, lo: int, n: int) -> str:
    if n == 16:
        if (hi, lo) == (1, 0):
            return "op"
        return f"funct{hi - lo + 1}"
    if (hi, lo) == (6, 0):
        return "opcode"
    if (hi, lo) == (14, 12):
        return "funct3"
    if lo >= 25:
        return f"funct{hi - lo + 1}"
    return {(11, 7): "rd", (19, 15): "rs1", (24, 20): "rs2"}.get((hi, lo), f"funct{hi - lo + 1}")


def _fixed_meaning(hi: int, lo: int, n: int, raw: int) -> str:
    if n == 32 and (hi, lo) == (6, 0):
        return OPCODES.get(raw >> 2 & 31, "")
    if n == 16 and (hi, lo) == (1, 0):
        return f"quadrant C{raw & 3}"
    return ""


def _operands(v: RVView, i, pc: int, before, after, log, mask: int, nxt: int | None,
              h: bool = False) -> None:
    name = i.base.name
    f = i.f
    regs0, regs1 = before[0], after[0]
    if fp_operands(name) and name not in _LOADS and name not in _STORES:
        return                                  # (the FPU's: the effects show it)

    def x(n: int) -> int:
        return regs0[n] & mask if n else 0

    sources = getattr(v, "sources", None) or {}

    def src(n: int) -> str:                     # (a bypass names its path)
        return sources.get(("r", n)) or _reg(n)

    rd = f.get("rd")
    if name in _ALU:
        a = Operand("rs1", x(f["rs1"]), src(f["rs1"])) if "rs1" in f else None
        if name == "auipc":
            a = Operand("pc", pc, "the pc")
        if "rs2" in f:
            b = Operand("rs2", x(f["rs2"]), src(f["rs2"]))
        elif "imm" in f:
            b = Operand("imm", f["imm"] & mask, "the immediate")
        elif "shamt" in f:
            b = Operand("shamt", f["shamt"], "the shift amount")
        else:
            b = None
        res = regs1[rd] & mask if rd else None
        v.alu = AluView(_ALU[name], a, b, res, res == 0 if res is not None else None)
    elif name in _LOADS or name in _STORES:
        size = _LOADS.get(name) or _STORES[name]
        base = x(f["rs1"])
        addr = (base + f.get("imm", 0)) & mask
        ent = next((e for e in log if e[1] == addr), None)
        tr = ent[-1] if ent is not None else None
        v.address = AddressView(Operand("rs1", base, src(f["rs1"])), f.get("imm", 0), addr,
                                size, "load" if name in _LOADS else "store",
                                tr.segment if tr is not None else "-",
                                (mode_name(before[4][_HSTATUS] >> 8 & 1, 1) + "-mode (HLV / "
                                 "HSV: hstatus.SPVP)") if name in _HLOADS or name in _HSTORES
                                else mode_name(*before[3], h) + "-mode",
                                bool(tr is not None and tr.mapped),
                                bool(tr is not None and tr.cca != 2), addr % size == 0,
                                tr.pa if tr is not None else None,
                                tr.cca if tr is not None else None,
                                tr.how if tr is not None else "")
    elif name.startswith("cbo.") and "_cmo" in v.__dict__:     # (unpriv. 19.6: the block)
        (block, pa, cca), note = v.__dict__.pop("_cmo")
        base = x(f["rs1"])
        v.address = AddressView(Operand("rs1", base, src(f["rs1"])), block - base, block, 64,
                                name, "-", mode_name(*before[3], h) + "-mode", pa != block,
                                cca != 2, True, pa, cca, note)
    elif name in _BRANCH:
        op, signed = _BRANCH[name]
        a, b = x(f["rs1"]), x(f["rs2"])
        target = (pc + f["imm"]) & mask
        v.branch = BranchView(f"{GPR_NAMES[f['rs1']]} {op} {GPR_NAMES[f['rs2']]}"
                              f"{' (signed)' if signed else ''}: 0x{a:x} {op} 0x{b:x}",
                              nxt == target and target != (pc + i.size) & mask, target)
    elif name in ("jal", "jalr"):
        target = (pc + f["imm"]) & mask if name == "jal" else (x(f["rs1"]) + f["imm"]) & mask & ~1
        v.branch = BranchView("always" if name == "jal" else
                              f"always (to {GPR_NAMES[f['rs1']]} + {f['imm']})", True, target,
                              link=(pc + i.size) & mask if rd else None)
        v.link_name = GPR_NAMES[rd] if rd else ""


def _effects(before, after, csr, flen: int) -> list[tuple[str, int, int]]:
    out = []
    for n in range(1, 32):
        if before[0][n] != after[0][n]:
            out.append((GPR_NAMES[n], before[0][n], after[0][n]))
    for n in range(32):
        if before[1][n] != after[1][n]:
            out.append((FPR_NAMES[n], before[1][n], after[1][n]))
    if before[2] != after[2]:
        out.append(("fcsr", before[2], after[2]))
    for (name, _attr), old, new in zip(_TRACKED, before[4], after[4]):
        if old != new:
            out.append((name, old, new))
    if csr is not None and csr[1] != csr[2] \
            and CSR_NAMES.get(csr[0]) not in {n for n, _a in _TRACKED}:
        out.append((CSR_NAMES.get(csr[0], f"csr 0x{csr[0]:03x}"), csr[1], csr[2]))
    if before[3][0] != after[3][0]:
        out.append(("priv", before[3][0], after[3][0]))
    if before[3][1] != after[3][1]:                 # (the virtualization mode: 21.1)
        out.append(("V", before[3][1], after[3][1]))
    vb, va = before[5], after[5]
    if vb is not None and va is not None:
        n = va[7]                               # (VLENB)
        for r in range(32):
            old, new = vb[0][r * n:(r + 1) * n], va[0][r * n:(r + 1) * n]
            if old != new:
                out.append((f"v{r}", int.from_bytes(old, "little"),
                            int.from_bytes(new, "little")))
        for k, name in ((1, "vl"), (2, "vtype"), (3, "vstart"), (4, "vxsat"), (5, "vxrm")):
            if vb[k] != va[k]:
                out.append((name, vb[k], va[k]))
    return out


class VectorLanes:
    """The elements a vector instruction changed (LF6): each written
    register's elements at the SEW after it, before -> after (the
    instruction pane's and ``format_view``'s element table)."""

    LIMIT = 32

    def __init__(self, before, after) -> None:
        self.sew = after[6]
        n, size = after[7], after[6] // 8
        self.rows: list[tuple[int, int, int, int]] = []
        for r in range(32):
            old, new = before[0][r * n:(r + 1) * n], after[0][r * n:(r + 1) * n]
            if old == new:
                continue
            for k in range(n // size):
                a = int.from_bytes(old[k * size:(k + 1) * size], "little")
                b = int.from_bytes(new[k * size:(k + 1) * size], "little")
                if a != b:
                    self.rows.append((r, k, a, b))

    def lines(self) -> list[str]:
        d = self.sew // 4
        out = [f"v{r}[{k}] e{self.sew}: 0x{a:0{d}x} -> 0x{b:0{d}x}"
               for r, k, a, b in self.rows[:self.LIMIT]]
        if len(self.rows) > self.LIMIT:
            out.append(f"... {len(self.rows) - self.LIMIT} more elements")
        return out
