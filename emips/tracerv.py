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
            ("stval", "stval"), ("sscratch", "sscratch"), ("satp", "satp"))
#: the base opcode map (unprivileged Table 70): bits 6:2 -> name
OPCODES = {0x00: "LOAD", 0x01: "LOAD-FP", 0x03: "MISC-MEM", 0x04: "OP-IMM", 0x05: "AUIPC",
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
}
_BRANCH = {"beq": ("==", False), "bne": ("!=", False), "blt": ("<", True),
           "bge": (">=", True), "bltu": ("<", False), "bgeu": (">=", False)}
_LOADS = {"lb": 1, "lh": 2, "lw": 4, "ld": 8, "lbu": 1, "lhu": 2, "lwu": 4, "flw": 4,
          "fld": 8}
_STORES = {"sb": 1, "sh": 2, "sw": 4, "sd": 8, "fsw": 4, "fsd": 8}


class RVView(InstrView):
    """The view of one RISC-V instruction (see the module docstring)."""

    rv = True

    def __getattr__(self, name: str):
        if name in InstrView._BUILT and "_raw" in self.__dict__:
            _build_rv(self)
            return self.__dict__[name]
        raise AttributeError(name)


def begin(m, pc: int) -> RVView:
    c = m.csrs
    before = (list(m.regs), list(m.fpr), m.fcsr, m.priv,
              tuple(getattr(c, attr) for _n, attr in _TRACKED))
    v = RVView(pc, 0, "", False, (m, pc, before))
    v.digits = 8 if m.xlen == 32 else 16
    v.csr_before = None
    m._mem_log = []
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
    after = (list(m.regs), list(m.fpr), m.fcsr, m.priv,
             tuple(getattr(c, attr) for _n, attr in _TRACKED))
    if i is not None:
        v.word, v.size, v.name = i.raw, i.size, i.row.name
    else:
        v.name = "(fetch)"
    v.exception = trap
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
        _operands(v, i, pc, before, after, log, mask, nxt)
    v.mem_reads = [(e[1], e[2], e[3]) for e in log if e[0] == "read"]
    v.mem_writes = [(e[1], e[2], e[3], e[4]) for e in log if e[0] == "write"]
    v.reg_writes = _effects(before, after, csr, flen)


def _format(i) -> str:
    if i.size == 2:
        return C_FORMATS.get(i.row.name, "C")
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
    if name == "csr":
        return CSR_NAMES.get(value, f"CSR 0x{value:03x}")
    if name == "rm":
        return RM_NAMES[value] or "reserved"
    if name in ("pred", "succ"):
        return "".join(c for b, c in zip((8, 4, 2, 1), "iorw") if value & b) or "-"
    return f"{name} = {value}"


def fp_operands(name: str) -> frozenset:
    """The operands of an F / D instruction (its 32-bit row's name) that
    are f registers (unprivileged ch. 20 / 21)."""
    if not name.startswith("f") or name in ("fence", "fence.i", "fence.tso"):
        return frozenset()
    if name in ("flw", "fld"):
        return frozenset({"rd"})
    if name in ("fsw", "fsd"):
        return frozenset({"rs2"})
    if name.startswith(("fmv.x.", "fclass.")) or name.startswith(("feq.", "flt.", "fle.")):
        return frozenset({"rs1", "rs2"})
    if name.startswith("fcvt."):
        to, frm = name.split(".")[1:3]
        out = set()
        if to in ("s", "d", "h", "q"):
            out.add("rd")
        if frm in ("s", "d", "h", "q"):
            out.add("rs1")
        return frozenset(out)
    if name.startswith("fmv.") and name.endswith(".x"):
        return frozenset({"rd"})
    return frozenset({"rd", "rs1", "rs2", "rs3"})


def _reg_of(name: str, n: int, fp: frozenset) -> str:
    return f"f{n} ({FPR_NAMES[n]})" if name in fp else _reg(n)


def fields(i) -> list[Field]:
    """Every bit field of the instruction (its own row: a compressed one's
    16 bits), MSB first; an immediate split over several fields names the
    bits of the immediate each holds."""
    row, raw, n = i.row, i.raw, 8 * i.size
    own = _fields(row, raw)
    fp = fp_operands(i.base.name)
    parts: list[tuple[int, int, str, str]] = []
    covered = 0
    last_of: dict[str, tuple[str, int]] = {}    # name -> (arg, segment) holding the total
    for a in row.args:
        names, segs, _w, _o = _ARGS[a]
        last_of[names[0]] = (a, len(segs) - 1)
    for a in row.args:
        names, segs, _width, offset = _ARGS[a]
        name = names[0]
        reg = name in ("rd", "rs1", "rs2", "rs3")
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
    for hi, lo in _fixed_runs(~covered & ((1 << n) - 1), n):
        parts.append((hi, lo, _fixed_name(hi, lo, n), _fixed_meaning(hi, lo, n, raw)))
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


def _operands(v: RVView, i, pc: int, before, after, log, mask: int, nxt: int | None) -> None:
    name = i.base.name
    f = i.f
    regs0, regs1 = before[0], after[0]
    if fp_operands(name) and name not in _LOADS and name not in _STORES:
        return                                  # (the FPU's: the effects show it)

    def x(n: int) -> int:
        return regs0[n] & mask if n else 0

    rd = f.get("rd")
    if name in _ALU:
        a = Operand("rs1", x(f["rs1"]), _reg(f["rs1"])) if "rs1" in f else None
        if name == "auipc":
            a = Operand("pc", pc, "the pc")
        if "rs2" in f:
            b = Operand("rs2", x(f["rs2"]), _reg(f["rs2"]))
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
        v.address = AddressView(Operand("rs1", base, _reg(f["rs1"])), f.get("imm", 0), addr,
                                size, "load" if name in _LOADS else "store",
                                tr.segment if tr is not None else "-",
                                {3: "M-mode", 1: "S-mode", 0: "U-mode"}[before[3]],
                                bool(tr is not None and tr.mapped),
                                bool(tr is not None and tr.cca != 2), addr % size == 0,
                                tr.pa if tr is not None else None,
                                tr.cca if tr is not None else None,
                                tr.how if tr is not None else "")
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
    if before[3] != after[3]:
        out.append(("priv", before[3], after[3]))
    return out
