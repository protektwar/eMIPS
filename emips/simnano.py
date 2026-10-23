"""The nanoMIPS32 machine (PLAN M23 phase 2).

``Machine(arch="nanomips")`` makes a :class:`NanoMachine`: a Release 6 core
that runs only nanoMIPS (MD01247 rev. 01.01) — 16 / 32 / 48-bit
instructions fetched at halfword addresses, no ISA mode (MD01251 3.5.4:
"the ISA Mode bit is 0"), no delay slots and no forbidden slots.  The
privileged architecture is MD01251 rev. 06.09's for a nanoMIPS core:
Config3.ISA = 1 (only nano/microMIPS), Config3.MMAR = 3 (Release 6
nanoMIPS32), BadInstr with a 16-bit instruction in bits 31:16 and a 48-bit
one's last halfword in BadInstrX (8, 3), no BadInstrP (Config3.BP = 0),
EPC = the instruction (BD always 0), an instruction fetch from an odd
address = AdEL (6.2.9: a variable-length ISA aligns to 16 bits).

Each instruction page of MD01247 has one executor (``NM.<page>``,
:data:`EXECUTORS`): it reads the arguments the format's decode steps left
in ``m._nm.f`` (:mod:`emips.nanomips`) and does the page's Operation.  The
CP0 / system instructions whose Operation is MIPS32 Release 6's run the
MIPS32 executor on a word with the same fields (:func:`_rr` and the
system rows of :data:`_PAGES`), so CP0
write masks, SRSCtl, Compare, the TLB, EJTAG and DVP / EVP live in one
place.  Choices: ``sim.DEVIATIONS["nanoMIPS"]``.
"""

from __future__ import annotations

from typing import Any, Callable

from . import cp0 as C
from . import nanomips as NM
from . import sim as S
from .isa import ISA, Decode, ISADecodeError

M32 = 0xFFFFFFFF

#: CP0 BadInstrX (MD01251 9.28)
BADINSTRX = (8, 3)
#: Config3.MMAR (20:18) = 3: Release 6 nanoMIPS32 (MD01251 Table 9.69)
CFG3_MMAR_SHIFT = 18
CFG3_BP, CFG3_BI = 1 << 27, 1 << 26
#: Config5 (MD01251 Table 9.71)
CFG5_NMS, CFG5_CRCP, CFG5_GI_SHIFT = 1 << 21, 1 << 18, 15
CFG5_XNP, CFG5_DEC, CFG5_MVH, CFG5_LLB = 1 << 13, 1 << 11, 1 << 5, 1 << 4

#: the EVA formats: entries of their own, so that sim's EVA gate (RI
#: without Config5.EVA, else CpU unless kernel / CU0, user rights) sees them
EVA_FORMS = frozenset({"LBE", "LBUE", "LHE", "LHUE", "LWE", "SBE", "SHE", "SWE", "CACHEE",
                       "PREFE", "SYNCIE", "LLE", "LLWPE", "SCE", "SCWPE"})
#: the EVA formats sim's gate leaves to their executor (CACHEE: an NMS
#: core without caches raises RI before the EVA and CP0 checks)
_OWN_EVA_GATE = frozenset({"CACHEE"})

#: each instruction page's family (sim's is_cti, the trace's CP0 snapshot)
_FAMILY = {
    **dict.fromkeys(("ADD", "ADDU", "AND", "NOR", "NOT", "OR", "XOR", "SLT", "SLTU", "SOV",
                     "MOVE", "MOVEP", "MOVN", "MOVZ", "CLO", "CLZ", "SEB", "SEH", "EXT", "INS",
                     "EXTW", "LSA", "ROTX", "CRC32B", "CRC32H", "CRC32W", "CRC32CB", "CRC32CH",
                     "CRC32CW", "SUB", "SUBU", "NOP"), "alu"),
    **dict.fromkeys(("ADDIU", "ADDIUPC", "ALUIPC", "ANDI", "ORI", "XORI", "SLTI", "SLTIU",
                     "SEQI", "LI", "LUI"), "imm"),
    **dict.fromkeys(("SLL", "SLLV", "SRL", "SRLV", "SRA", "SRAV", "ROTR", "ROTRV"), "shift"),
    **dict.fromkeys(("MUL", "MUH", "MULU", "MUHU", "DIV", "DIVU", "MOD", "MODU"), "muldiv"),
    **dict.fromkeys(("BBEQZC", "BBNEZC", "BEQC", "BEQIC", "BEQZC", "BGEC", "BGEIC", "BGEIUC",
                     "BGEUC", "BLTC", "BLTIC", "BLTIUC", "BLTUC", "BNEC", "BNEIC", "BNEZC"),
                    "branch_compact"),
    **dict.fromkeys(("BALC", "BC", "BALRSC", "BRSC", "JALRC", "JALRC.HB", "JRC",
                     "MOVE.BALC"), "jump_compact"),
    **dict.fromkeys(("LB", "LBU", "LH", "LHU", "LW", "LBX", "LBUX", "LHX", "LHUX", "LHXS",
                     "LHUXS", "LWX", "LWXS", "LWM", "LWPC", "UALH", "UALWM", "LL", "LBE", "LBUE",
                     "LHE", "LHUE", "LWE", "RESTORE"), "load"),
    **dict.fromkeys(("SB", "SH", "SW", "SBX", "SHX", "SWX", "SHXS", "SWXS", "SWM", "SWPC",
                     "UASH", "UASWM", "SC", "SBE", "SHE", "SWE", "SAVE"), "store"),
    **dict.fromkeys(("TEQ", "TNE", "BREAK", "SYSCALL", "SIGRIE", "SDBBP"), "trap"),
    **dict.fromkeys(("MFC0", "MTC0", "MFHC0", "MTHC0", "DI", "EI", "ERET", "DERET", "WAIT",
                     "RDPGPR", "WRPGPR", "TLBP", "TLBR", "TLBWI", "TLBWR", "TLBINV", "TLBINVF",
                     "DVP", "EVP", "GINVI", "GINVT", "RDHWR"), "cop0"),
    **dict.fromkeys(("CACHE", "PREF", "SYNC", "SYNCI", "EHB", "PAUSE"), "cache"),
    **dict.fromkeys(NM.DSP_NAMES - {"BPOSGE32C"}, "dsp"),
    **dict.fromkeys(("DMT", "EMT", "DVPE", "EVPE", "FORK", "YIELD", "MFTR", "MTTR"), "mt"),
    "BPOSGE32C": "branch_compact",
}


def entry_name(row: NM.Row) -> str:
    """The Decode entry of a format: ``NM.<page>`` (``NM.<form>`` for an
    EVA format)."""
    return f"NM.{row.form if row.form in EVA_FORMS else row.name}"


def _entry(name: str, page: str) -> dict:
    e = {"name": name, "mnemonic": page.lower(), "family": _FAMILY[page],
         "status": "core", "pic32": False, "isa": "nanomips",
         "decode": {"lead": True}, "releases": ["R6"]}
    if page in NM.DSP_NAMES:                    # (sim's DSP State Disabled gate)
        e["ase"] = "dsp"
    return e


#: the Decode entries (sim.Decode(raw, ENTRIES[entry_name(row)]))
ENTRIES = {entry_name(r): _entry(entry_name(r), r.name) for r in NM.ROWS}
assert set(_FAMILY) == NM.NAMES, set(_FAMILY) ^ NM.NAMES

#: what sim's _dispatch gates: CpU unless kernel mode or Status.CU0
#: (MD01251 3.5.3; the instructions whose page raises RI first check
#: themselves: TLB*, GINV*, DVP / EVP)
PRIVILEGED = frozenset({"NM.MFC0", "NM.MTC0", "NM.MFHC0", "NM.MTHC0", "NM.DI", "NM.EI",
                        "NM.ERET", "NM.DERET", "NM.WAIT", "NM.RDPGPR", "NM.WRPGPR",
                        "NM.DMT", "NM.EMT", "NM.DVPE", "NM.EVPE", "NM.MFTR", "NM.MTTR"})
EVA = frozenset(f"NM.{f}" for f in EVA_FORMS - _OWN_EVA_GATE)


def _sx32(v: int) -> int:
    v &= M32
    return v - (1 << 32) if v >> 31 else v


#: decoded instructions by (raw, size): decoding is a pure function of them
_DECODED: dict[tuple[int, int], NM.NM] = {}


def decode(raw: int, size: int) -> NM.NM:
    key = (raw, size)
    i = _DECODED.get(key)
    if i is None:
        if len(_DECODED) > 65536:
            _DECODED.clear()
        i = _DECODED[key] = NM.decode_raw(raw, size)
    return i


class NanoMachine(S.Machine):
    """A nanoMIPS32 Release 6 core (see the module docstring).  ``nms``:
    the nanoMIPS subset (Config5.NMS: the formats MD01247 marks "not in
    NMS" are Reserved Instructions), ``crc``: the CRC32 instructions
    (Config5.CRCP), ``ginv``: Config5.GI (0 none, 2 GINVI, 3 GINVI and
    GINVT)."""

    arch = "nanomips"
    nano = True
    names = "p32"

    def __init__(self, isa: ISA | None = None, memory: Any = None, pc: int = 0, *,
                 release: str | None = None, compressed: str | None = None,
                 dsp: int | None = None, mt: bool = False, arch: str = "nanomips",
                 fpu: bool = False, nms: bool = False, crc: bool = False, ginv: int = 0,
                 **kwargs: Any) -> None:
        if release not in (None, "R6"):
            raise ValueError("nanoMIPS is Release 6 (MD01247 ch. 1)")
        if compressed is not None:
            raise ValueError("a nanoMIPS core has no compressed ISA mode (MD01251 3.5.4)")
        if fpu:
            raise ValueError("MD01247 defines no nanoMIPS FPU instructions (the CP1 cells "
                             "of its opcode map are reserved)")
        if ginv not in (0, 2, 3):
            raise ValueError("ginv is Config5.GI: 0, 2 or 3")
        level = 2 if dsp else 0                 # (Revision 3: DSPP = DSP2P = 1)
        super().__init__(isa if isa is not None else ISA.for_release("R6", level), memory,
                         pc, release="R6", fpu=False, dsp=level, mt=mt, **kwargs)
        #: Config5.NMS, CRCP, GI
        self.nms, self.crc, self.ginv = nms, crc, ginv
        self.isa_mode = 0
        self.mips32 = False
        #: the nanoMIPS instruction being executed (its NM.* executor)
        self._nm: NM.NM | None = None
        #: its size (BadInstr / BadInstrX), the size of the LL's access
        self._insn_size = 4
        self.lr_size = 4
        cfg3 = self.cp0[C.CONFIG3]["value"] & ~(3 << C.CFG3_ISA_SHIFT | 7 << CFG3_MMAR_SHIFT
                                                | CFG3_BP)
        self.cp0[C.CONFIG3]["value"] = (cfg3 | 1 << C.CFG3_ISA_SHIFT | 3 << CFG3_MMAR_SHIFT
                                        | C.CFG3_ISAONEXC | CFG3_BI)
        cfg5 = self.cp0[(16, 5)]
        cfg5["value"] |= (CFG5_LLB | CFG5_MVH | CFG5_DEC | (CFG5_NMS if nms else 0)
                          | (CFG5_CRCP if crc else 0) | ginv << CFG5_GI_SHIFT)
        if not nms:                             # (MD01251 9.28: none with NMS)
            self.cp0[BADINSTRX] = {"name": "BadInstrX", "value": 0}
            self._wmask[BADINSTRX] = 0

    def _fetch_nm(self, pc: int) -> tuple[int, int, Decode]:
        """Fetch and decode the instruction at ``pc`` (2, 4 or 6 bytes):
        (size, its encoding, its Decode).  An odd PC or a halfword outside
        the mode's segments is AdEL (BadVAddr = that address)."""
        if pc & 1 or not self.address_ok(pc):
            raise S._Arch(C.ADEL, badvaddr=pc)
        h = self.fetch_half(pc)
        size = NM.size_of(h)
        raw = h
        for k in range(2, size, 2):
            a = (pc + k) & M32
            if not self.address_ok(a):
                raise S._Arch(C.ADEL, badvaddr=a)
            raw = raw << 16 | self._second_half(a)
        self._insn_raw = raw
        self._insn_size = size
        try:
            i = decode(raw, size)
        except ISADecodeError as exc:
            if self.exceptions == "raise":
                raise
            raise S._Arch(C.RI) from exc
        if self.nms and not i.row.nms:          # "not available in NMS" (MD01247)
            raise S._Arch(C.RI)
        if i.row.need == "dsp" and not self.dsp:     # Config3.DSPP = 0 (MD01249 6.2.1)
            raise S._Arch(C.RI)
        if i.row.need == "mt" and self.mt is None:   # no MT ASE (MD01255 5.1)
            raise S._Arch(C.RI)
        self._nm = i
        return size, raw, Decode(raw, ENTRIES[entry_name(i.row)])

    def fetch_compressed(self, pc: int) -> tuple[int, int, Decode, Any]:
        """The pipelines' fetch (emips.pipeline.fetch_nano): (size, the
        encoding, its Decode, the decoded instruction), as :meth:`step`
        fetches."""
        size, raw, dec = self._fetch_nm(pc)
        return size, raw, dec, self._nm

    def enter_nm(self, insn: NM.NM | None, raw: int | None, size: int) -> None:
        """The pipelines, before an instruction executes or takes its
        exception (they fetch ahead): the instruction its executor reads
        (``_nm``), its encoding and size (BadInstr / BadInstrX)."""
        self._nm = insn
        self._insn_raw = raw
        self._insn_size = size

    def _bad_instr(self, exc: Any, pc: int, word: int | None, in_slot: bool) -> None:
        """BadInstr / BadInstrX (MD01251 9.26 / 9.28, MD01247 4.14): a
        16-bit instruction in BadInstr 31:16, a 32-bit one whole, a 48-bit
        one's first 32 bits in BadInstr and its last halfword in BadInstrX
        31:16 (an NMS core has no BadInstrX).  No BadInstrP (no delay
        slots)."""
        if exc.code in self._NO_BADINSTR or word is None or self._insn_raw is None:
            return
        raw, size = self._insn_raw, self._insn_size
        if size == 2:
            self._cp0_set(C.BADINSTR, raw << 16)
        elif size == 4:
            self._cp0_set(C.BADINSTR, raw)
        else:
            self._cp0_set(C.BADINSTR, raw >> 16)
            if BADINSTRX in self.cp0:
                self._cp0_set(BADINSTRX, (raw & 0xFFFF) << 16)


# ---------------------------------------------------------------- helpers

def _f(m: Any) -> dict:
    return m._nm.f


def _r(m: Any, n: int) -> int:
    return m.read_reg(n) & M32


def _s(m: Any, n: int) -> int:
    return _sx32(m.read_reg(n))


def _next(m: Any) -> int:
    """CPU.next_pc: the instruction after this one."""
    return (m.pc + m._nm.size) & M32


def _go(m: Any, offset: int) -> None:
    """``CPU.next_pc = effective_address(CPU.next_pc, offset)`` (no delay
    slot: sim's _redirect)."""
    m._redirect = (_next(m) + offset) & M32


def _user(m: Any) -> bool:
    """EffectiveKSU() == 2 (MD01247 4.13): User Mode."""
    status = m.status
    return C.ksu(status) == 2 and not status & (C.ST_EXL | C.ST_ERL) and not m._dm


def _load(m: Any, va: int, n: int, signed: bool, *, aligned: bool = False) -> int:
    v = m.load(m.data_address(va & M32, n, store=False, aligned=aligned), n)
    return v - (1 << 8 * n) if signed and v >> (8 * n - 1) else v


def _store(m: Any, va: int, value: int, n: int, *, aligned: bool = False) -> None:
    m.store(m.data_address(va & M32, n, store=True, aligned=aligned),
            value & ((1 << 8 * n) - 1), n)


def _cp0_ok(m: Any) -> None:
    """IsCoprocessor0Enabled(), else Coprocessor Unusable 0 (MD01251
    3.5.3: kernel / debug mode or Status.CU0)."""
    if not (m.kernel_mode or m.status & C.ST_CU0):
        raise S._Arch(C.CPU, ce=0)


# ---------------------------------------------------------------- ALU

def _add(m: Any) -> None:
    f = _f(m)
    total = _s(m, f["rs"]) + _s(m, f["rt"])
    if not -(1 << 31) <= total < 1 << 31:
        raise S._Arch(C.OV)
    m.write_reg(f["rd"], total)


def _sub(m: Any) -> None:
    f = _f(m)
    total = _s(m, f["rs"]) - _s(m, f["rt"])
    if not -(1 << 31) <= total < 1 << 31:
        raise S._Arch(C.OV)
    m.write_reg(f["rd"], total)


def _sov(m: Any) -> None:
    f = _f(m)
    total = _s(m, f["rs"]) + _s(m, f["rt"])
    m.write_reg(f["rd"], int(not -(1 << 31) <= total < 1 << 31))


def _rd3(op: Callable[[int, int], int], rd: str = "rd", a: str = "rs",
         b: str = "rt") -> Callable[[Any], None]:
    def run(m: Any) -> None:
        f = _f(m)
        m.write_reg(f[rd], op(_r(m, f[a]), _r(m, f[b])))
    return run


def _imm(op: Callable[[int, int], int], value: str) -> Callable[[Any], None]:
    def run(m: Any) -> None:
        f = _f(m)
        m.write_reg(f["rt"], op(_r(m, f["rs"]), f[value]))
    return run


def _addiu(m: Any) -> None:
    f = _f(m)
    m.write_reg(f["rt"], _r(m, f["rs"]) + f["imm"])


def _addiupc(m: Any) -> None:
    f = _f(m)
    m.write_reg(f["rt"], _next(m) + f["s"])


def _aluipc(m: Any) -> None:
    f = _f(m)
    m.write_reg(f["rt"], (_next(m) + f["offset"]) & ~0xFFF)


def _rot(v: int, n: int) -> int:
    v &= M32
    return ((v >> n) | (v << (32 - n))) & M32 if n else v


def _count(m: Any, ones: bool) -> None:
    f = _f(m)
    v = _r(m, f["rs"]) ^ (M32 if ones else 0)
    m.write_reg(f["rt"], 32 - v.bit_length())


def _ext(m: Any) -> None:
    """EXT: pos + size > 32 is UNPREDICTABLE: the bits there are (zeros
    above bit 31)."""
    f = _f(m)
    m.write_reg(f["rt"], (_r(m, f["rs"]) >> f["pos"]) & ((1 << f["size"]) - 1))


def _ins(m: Any) -> None:
    """INS: size < 1 is UNPREDICTABLE: Reserved Instruction (QEMU)."""
    f = _f(m)
    pos, size = f["pos"], f["size"]
    if size < 1:
        raise S._Arch(C.RI)
    mask = ((1 << size) - 1) << pos
    m.write_reg(f["rt"], (_r(m, f["rt"]) & ~mask) | ((_r(m, f["rs"]) << pos) & mask))


def _extw(m: Any) -> None:
    f = _f(m)
    tmp = _r(m, f["rt"]) << 32 | _r(m, f["rs"])
    m.write_reg(f["rd"], tmp >> f["shift"])


def _lsa(m: Any) -> None:
    f = _f(m)
    m.write_reg(f["rd"], (_r(m, f["rs"]) << f["u2"]) + _r(m, f["rt"]))


def _rotx(m: Any) -> None:
    """ROTX: the Operation's five barrel-shifter stages, bit for bit."""
    f = _f(m)
    shift, shiftx, stripe = f["shift"], f["shiftx"], f["stripe"]
    rs = _r(m, f["rs"])
    t0 = rs << 32 | rs

    def bit(v: int, i: int) -> int:
        return (v >> i) & 1

    def stage(src: int, n: int, step: int, sel: int, use_stripe: bool = False) -> int:
        dst = src
        for i in range(n):
            s = shift if i & sel else shiftx
            if use_stripe and stripe and not i & 0b00100:
                s = ~s
            if (s >> step.bit_length() - 1) & 1:
                dst = (dst & ~(1 << i)) | (bit(src, i + step) << i)
        return dst
    t1 = stage(t0, 47, 16, 0b01000, True)
    t2 = stage(t1, 39, 8, 0b00100)
    t3 = stage(t2, 35, 4, 0b00010)
    t4 = stage(t3, 33, 2, 0b00001)
    t5 = t4
    for i in range(32):
        if shift & 1:
            t5 = (t5 & ~(1 << i)) | (bit(t4, i + 1) << i)
    m.write_reg(f["rt"], t5 & M32)


def _crc32(nbits: int, poly: int) -> Callable[[Any], None]:
    """CRC32B / H / W, CRC32CB / CH / CW (MD01247 4.5: the reflected CRC of
    the low ``nbits`` of rs into rt); Config5.CRCP = 0: RI."""
    def run(m: Any) -> None:
        if not m.crc:
            raise S._Arch(C.RI)
        f = _f(m)
        value, message = _r(m, f["rt"]), _r(m, f["rs"]) & ((1 << nbits) - 1)
        value ^= message
        for _ in range(nbits):
            value = (value >> 1) ^ (poly if value & 1 else 0)
        m.write_reg(f["rt"], value)
    return run


def _movep(m: Any) -> None:
    """MOVEP: an output that is also an input is UNPREDICTABLE: both
    inputs are read first."""
    f = _f(m)
    a, b = m.read_reg(f["src1"]), m.read_reg(f["src2"])
    m.write_reg(f["dst1"], a)
    m.write_reg(f["dst2"], b)


# ---------------------------------------------------------------- branches

def _branch(cond: Callable[[Any, dict], bool]) -> Callable[[Any], None]:
    def run(m: Any) -> None:
        f = _f(m)
        if cond(m, f):
            _go(m, f["offset"])
    return run


def _bbit(want: int) -> Callable[[Any], None]:
    """BBEQZC / BBNEZC: bit 32-63 needs 64-bit operations
    (Are64BitOperationsEnabled(): not User Mode — Status.PX / UX read 0
    here), and is then the sign bit of the 32-bit register."""
    def run(m: Any) -> None:
        f = _f(m)
        bit = f["bit"]
        if bit >= 32 and _user(m):
            raise S._Arch(C.RI)
        if (_s(m, f["rt"]) >> bit) & 1 == want:
            _go(m, f["offset"])
    return run


def _balc(m: Any) -> None:
    m.write_reg(31, _next(m))
    _go(m, _f(m)["offset"])


def _brsc(link: bool) -> Callable[[Any], None]:
    def run(m: Any) -> None:
        f = _f(m)
        address = (_next(m) + (_r(m, f["rs"]) << 1)) & M32
        if link:
            m.write_reg(f["rt"], _next(m))
        m._redirect = address
    return run


def _jalrc(m: Any) -> None:
    f = _f(m)
    address = _r(m, f["src"])
    m.write_reg(f["dst"], _next(m))
    m._redirect = address


def _jalrc_hb(m: Any) -> None:
    f = _f(m)
    address = _r(m, f["rs"])
    m.write_reg(f["rt"], _next(m))
    m._redirect = address


def _jrc(m: Any) -> None:
    m._redirect = _r(m, _f(m)["rt"])


def _move_balc(m: Any) -> None:
    f = _f(m)
    m.write_reg(f["rd"], m.read_reg(f["rt"]))
    m.write_reg(31, _next(m))
    _go(m, f["offset"])


# ---------------------------------------------------------------- loads / stores

def _ld(n: int, signed: bool) -> Callable[[Any], None]:
    def run(m: Any) -> None:
        f = _f(m)
        m.write_reg(f["rt"], _load(m, _r(m, f["rs"]) + f["offset"], n, signed))
    return run


def _st(n: int) -> Callable[[Any], None]:
    def run(m: Any) -> None:
        f = _f(m)
        _store(m, _r(m, f["rs"]) + f["offset"], _r(m, f["rt"]), n)
    return run


def _ldx(n: int, signed: bool, scale: int = 0) -> Callable[[Any], None]:
    """LBX ... / LHXS / LWXS: va = (GPR[rs] << scale) + GPR[rt]."""
    def run(m: Any) -> None:
        f = _f(m)
        va = (_r(m, f["rs"]) << scale) + _r(m, f["rt"])
        m.write_reg(f["rd"], _load(m, va, n, signed))
    return run


def _stx(n: int, scale: int = 0) -> Callable[[Any], None]:
    def run(m: Any) -> None:
        f = _f(m)
        _store(m, (_r(m, f["rs"]) << scale) + _r(m, f["rt"]), _r(m, f["rd"]), n)
    return run


def _lwpc(m: Any) -> None:
    f = _f(m)
    m.write_reg(f["rt"], _load(m, _next(m) + f["offset"], 4, True))


def _swpc(m: Any) -> None:
    f = _f(m)
    _store(m, _next(m) + f["offset"], _r(m, f["rt"]), 4)


def _list(rt: int, count: int, gp: int = 0) -> list[int]:
    return NM.reglist(rt, count, gp)


def _lwm(m: Any) -> None:
    """LWM / UALWM: count words to rt, rt + 1 ... ($31 wraps to $16) from
    rs + offset, rs + offset + 4 ...; the base is read once (a base
    register loaded before the last word is UNPREDICTABLE)."""
    f = _f(m)
    base = _r(m, f["rs"])
    for i, reg in enumerate(_list(f["rt"], f["count"])):
        m.write_reg(reg, _load(m, base + f["offset"] + 4 * i, 4, True))


def _swm(m: Any) -> None:
    """SWM / UASWM: rt = 0 stores $0 count times."""
    f = _f(m)
    base = _r(m, f["rs"])
    regs = [0] * f["count"] if f["rt"] == 0 else _list(f["rt"], f["count"])
    for i, reg in enumerate(regs):
        _store(m, base + f["offset"] + 4 * i, _r(m, reg), 4)


def _save(m: Any) -> None:
    """SAVE: the registers to $29 - 4, - 8 ... (an unaligned one is AdES),
    then $29 -= u; gp = 1 on an NMS core: RI."""
    f = _f(m)
    if f["gp"] and m.nms:
        raise S._Arch(C.RI)
    sp = _r(m, 29)
    for i, reg in enumerate(_list(f["rt"], f["count"], f["gp"])):
        _store(m, sp - 4 * (i + 1), _r(m, reg), 4, aligned=True)
    m.write_reg(29, sp - f["u"])


def _restore(m: Any) -> None:
    """RESTORE / RESTORE.JRC: from $29 + u - 4, - 8 ...; $29 += u; JRC:
    to the (restored) $31; gp = 1 on an NMS core: RI."""
    f = _f(m)
    if f["gp"] and m.nms:
        raise S._Arch(C.RI)
    sp = _r(m, 29)
    for i, reg in enumerate(_list(f["rt"], f["count"], f["gp"])):
        m.write_reg(reg, _load(m, sp + f["u"] - 4 * (i + 1), 4, True, aligned=True))
    m.write_reg(29, sp + f["u"])
    if f["jr"]:
        m._redirect = _r(m, 31)


def _ll(m: Any) -> None:
    """LL / LLE / LLWP / LLWPE (MD01247): aligned to nbytes; LLWP's words
    are at va (rt) and va + 4 (ru) in either byte order; rt = ru is
    UNPREDICTABLE (ru wins)."""
    f = _f(m)
    n = f["nbytes"]
    va = (_r(m, f["rs"]) + f["offset"]) & M32
    addr = m.data_address(va, n, store=False, atomic=True)
    m._linked = True
    try:
        if n == 4:
            m.write_reg(f["rt"], m.load(addr, 4))
        else:
            w0, w1 = m.load(addr, 4), m.load((addr + 4) & M32, 4)
            m.write_reg(f["rt"], w0)
            m.write_reg(f["ru"], w1)
    finally:
        m._linked = False
    m.lr_addr, m.lr_valid = addr, True
    m.lr_size = n
    if m.system is not None:
        m.lr_pa = m._peek(addr)[0]


def _sc(m: Any) -> None:
    """SC / SCE / SCWP / SCWPE: store (rt; SCWP rt at va, ru at va + 4)
    only while LLbit is set for this address and size; rt = 1 / 0; LLbit
    cleared."""
    f = _f(m)
    n = f["nbytes"]
    va = (_r(m, f["rs"]) + f["offset"]) & M32
    addr = m.data_address(va, n, store=True, atomic=True)
    ok = m.lr_valid and m.lr_addr == addr and getattr(m, "lr_size", 4) == n
    if ok:
        m.store(addr, _r(m, f["rt"]), 4, gather=False, atomic=True)
        if n == 8:
            m.store((addr + 4) & M32, _r(m, f["ru"]), 4, gather=False, atomic=True)
    m.write_reg(f["rt"], int(ok))
    m.lr_valid = False


# ---------------------------------------------------------------- system

def _mfc0(high: bool) -> Callable[[Any], None]:
    """MFC0 / MFHC0 (CP0 checked by sim's _dispatch): a register this core
    does not have reads 0 (MD01247 MFC0); no register has high bits here
    (no XPA): MFHC0 reads 0."""
    def run(m: Any) -> None:
        f = _f(m)
        key = (f["c0s"], f["sel"])
        m.write_reg(f["rt"], 0 if high or key not in m.cp0 else m.cp0_read(*key))
    return run


def _mtc0(high: bool) -> Callable[[Any], None]:
    """MTC0 / MTHC0: a register this core does not have ignores the write;
    MTHC0 writes nothing (no high bits)."""
    def run(m: Any) -> None:
        f = _f(m)
        key = (f["c0s"], f["sel"])
        if not high and key in m.cp0:
            m.mtc0(*key, m.read_reg(f["rt"]))
    return run


def _eret(m: Any) -> None:
    """ERET / ERETNC: ERETNC keeps LLbit (and needs Config5.LLB: 1)."""
    keep = m.lr_valid
    S._EXECUTORS["ERET"](m, 0)
    if _f(m)["nc"]:
        m.lr_valid = keep


def _tlb(op: str, inv: bool = False) -> Callable[[Any], None]:
    """TLBP / TLBR / TLBWI / TLBWR / TLBINV / TLBINVF: RI without a TLB
    (or Config4.IE < 2), then CpU (MD01247: RI is checked first)."""
    def run(m: Any) -> None:
        fn = getattr(m.mmu, op, None)
        if fn is None or inv and not m.mmu.invalidate:
            raise S._Arch(C.RI)
        _cp0_ok(m)
        fn(m)
    return run


def _ginv(level: int) -> Callable[[Any], None]:
    """GINVI (Config5.GI >= 2) / GINVT (GI = 3): RI without, then CpU; the
    caches / TLBs eMIPS has are invalidated by the Machine's own ops."""
    def run(m: Any) -> None:
        if m.ginv < level:
            raise S._Arch(C.RI)
        _cp0_ok(m)
        if level == 2 and m.caches is not None and m.caches.i is not None:
            m.caches.i.invalidate_all()
        elif level == 3:
            inv = getattr(m.mmu, "tlbinvf", None)
            if inv is not None:
                inv(m)
    return run


def _vp(name: str) -> Callable[[Any], None]:
    """DVP / EVP: a NOP without Config5.VP (MD01247: before any CpU)."""
    run = S._EXECUTORS[name]

    def go(m: Any) -> None:
        if m.vpcore is None:
            return
        _cp0_ok(m)
        run(m, _f(m)["rt"] << 16)
    return go


def _rdhwr(m: Any) -> None:
    """RDHWR rt, hs, sel (MD01247): without CP0 access only the HWREna
    registers; sel only for hs 4 (a performance counter, CP0 25.sel); 5
    XNP (0: paired LL/SC here), 29 UserLocal."""
    f = _f(m)
    hs, sel = f["hs"], f["sel"]
    cp0 = m.kernel_mode or m.status & C.ST_CU0
    if not cp0 and not m.cp0_read(*C.HWRENA) >> hs & 1:
        raise S._Arch(C.RI)
    if sel and hs != 4:
        raise S._Arch(C.RI)
    if hs == 0:
        v = m.cp0_read(*C.EBASE) & 0x3FF
    elif hs == 1:
        v = m.caches.synci_step if m.caches is not None else 0
    elif hs == 2:
        v = m.cp0_read(*C.COUNT)
    elif hs == 3:
        v = m.ccres
    elif hs == 4:
        if not m.cp0_read(*C.CONFIG1) >> 4 & 1 or (25, sel) not in m.cp0:
            raise S._Arch(C.RI)
        v = m.cp0_read(25, sel)
    elif hs == 5:
        v = int(bool(m.cp0_read(16, 5) & CFG5_XNP))
    elif hs == 29:
        v = m.cp0_read(*C.USERLOCAL)
    else:
        raise S._Arch(C.RI)
    m.write_reg(f["rt"], v)


def _trap(eq: bool) -> Callable[[Any], None]:
    def run(m: Any) -> None:
        f = _f(m)
        if (_r(m, f["rs"]) == _r(m, f["rt"])) == eq:
            raise S._Arch(C.TR, info={"code": f["code"]})
    return run


def _syscall(m: Any) -> None:
    code = _f(m)["code"]
    if m.syscall_handler is not None and m.syscall_handler(m, code) is not False:
        return
    raise S._Arch(C.SYS, info={"code": code})


def _sigrie(m: Any) -> None:
    raise S._Arch(C.RI)


def _cache(m: Any) -> None:
    """CACHE / CACHEE (MD01247): an NMS core without caches raises RI, then
    CACHEE needs Config5.EVA (RI), then CP0 access (CpU)."""
    f = _f(m)
    if m.nms and m.caches is None:
        raise S._Arch(C.RI)
    eva = m._nm.row.form == "CACHEE"
    if eva and not getattr(m.mmu, "eva", False):
        raise S._Arch(C.RI)
    _cp0_ok(m)
    m._eva = eva
    try:
        m.cache_op(f["op"], (_r(m, f["rs"]) + f["offset"]) & M32)
    finally:
        m._eva = False


def _pref(m: Any) -> None:
    """PREF / PREFE: hints 24-30 are NOPs in nanoMIPS (MD01247 PREF: not
    RI as MIPS Release 6, nor the 24K's Nudge); never an address or TLB
    exception."""
    f = _f(m)
    if 24 <= f["hint"] <= 30:
        return
    m.prefetch(f["hint"], (_r(m, f["rs"]) + f["offset"]) & M32)


def _synci(m: Any) -> None:
    f = _f(m)
    m.synci((_r(m, f["rs"]) + f["offset"]) & M32)


def _nop(m: Any) -> None:
    return None


def _wrap(fn: Callable[[Any], Any]) -> S.Executor:
    def run(m: Any, w: int) -> int | None:
        fn(m)
        return None
    run.__name__ = getattr(fn, "__name__", "run")
    return run


def _rr(name: str, rd: str = "rd", rs: str = "rs", rt: str = "rt") -> Callable[[Any], None]:
    """A MIPS32 Release 6 three-register executor (MUL ... MODU: the same
    Operation, sim's choices for a zero divisor)."""
    run = S._EXECUTORS[name]

    def go(m: Any) -> None:
        f = _f(m)
        run(m, f[rs] << 21 | f[rt] << 16 | f[rd] << 11)
    return go


_PAGES: dict[str, Callable[[Any], Any]] = {
    "ADD": _add, "SUB": _sub, "SOV": _sov,
    "ADDU": _rd3(lambda a, b: a + b, "dst", "src1", "src2"),
    "SUBU": _rd3(lambda a, b: a - b),
    "AND": _rd3(lambda a, b: a & b), "OR": _rd3(lambda a, b: a | b),
    "XOR": _rd3(lambda a, b: a ^ b), "NOR": _rd3(lambda a, b: ~(a | b)),
    "SLT": _rd3(lambda a, b: int(_sx32(a) < _sx32(b))),
    "SLTU": _rd3(lambda a, b: int(a < b)),
    "SLLV": _rd3(lambda a, b: a << (b & 31)),
    "SRLV": _rd3(lambda a, b: a >> (b & 31)),
    "SRAV": _rd3(lambda a, b: _sx32(a) >> (b & 31)),
    "ROTRV": _rd3(lambda a, b: _rot(a, b & 31)),
    "MOVN": lambda m: m.write_reg(_f(m)["rd"], m.read_reg(_f(m)["rs"]))
    if _r(m, _f(m)["rt"]) else None,
    "MOVZ": lambda m: m.write_reg(_f(m)["rd"], m.read_reg(_f(m)["rs"]))
    if not _r(m, _f(m)["rt"]) else None,
    "NOT": lambda m: m.write_reg(_f(m)["rt"], ~_r(m, _f(m)["rs"])),
    "MOVE": lambda m: m.write_reg(_f(m)["rt"], m.read_reg(_f(m)["rs"])),
    "MOVEP": _movep,
    "MUL": _rr("MUL@R6", "dst", "src1", "src2"), "MUH": _rr("MUH"), "MULU": _rr("MULU"),
    "MUHU": _rr("MUHU"), "DIV": _rr("DIV@R6"), "DIVU": _rr("DIVU@R6"), "MOD": _rr("MOD"),
    "MODU": _rr("MODU"),
    "CLO": lambda m: _count(m, True), "CLZ": lambda m: _count(m, False),
    "SEB": lambda m: m.write_reg(_f(m)["rt"], (_r(m, _f(m)["rs"]) & 0xFF ^ 0x80) - 0x80),
    "SEH": lambda m: m.write_reg(_f(m)["rt"],
                                 (_r(m, _f(m)["rs"]) & 0xFFFF ^ 0x8000) - 0x8000),
    "EXT": _ext, "INS": _ins, "EXTW": _extw, "LSA": _lsa, "ROTX": _rotx,
    "CRC32B": _crc32(8, 0xEDB88320), "CRC32H": _crc32(16, 0xEDB88320),
    "CRC32W": _crc32(32, 0xEDB88320), "CRC32CB": _crc32(8, 0x82F63B78),
    "CRC32CH": _crc32(16, 0x82F63B78), "CRC32CW": _crc32(32, 0x82F63B78),
    "ADDIU": _addiu, "ADDIUPC": _addiupc, "ALUIPC": _aluipc,
    "ANDI": _imm(lambda a, u: a & u, "u"), "ORI": _imm(lambda a, u: a | u, "u"),
    "XORI": _imm(lambda a, u: a ^ u, "u"),
    "SLTI": _imm(lambda a, u: int(_sx32(a) < u), "u"),
    "SLTIU": _imm(lambda a, u: int(a < u), "u"),
    "SEQI": _imm(lambda a, u: int(a == u), "u"),
    "LI": lambda m: m.write_reg(_f(m)["rt"], _f(m)["s"]),
    "LUI": lambda m: m.write_reg(_f(m)["rt"], _f(m)["imm"]),
    "SLL": _imm(lambda a, s: a << s, "shift"), "SRL": _imm(lambda a, s: a >> s, "shift"),
    "SRA": _imm(lambda a, s: _sx32(a) >> s, "shift"),
    "ROTR": _imm(lambda a, s: _rot(a, s), "shift"),
    # branches and jumps (no delay slot)
    "BC": lambda m: _go(m, _f(m)["offset"]), "BALC": _balc,
    "BEQC": _branch(lambda m, f: _r(m, f["rs"]) == _r(m, f["rt"])),
    "BNEC": _branch(lambda m, f: _r(m, f["rs"]) != _r(m, f["rt"])),
    "BGEC": _branch(lambda m, f: _s(m, f["rs"]) >= _s(m, f["rt"])),
    "BLTC": _branch(lambda m, f: _s(m, f["rs"]) < _s(m, f["rt"])),
    "BGEUC": _branch(lambda m, f: _r(m, f["rs"]) >= _r(m, f["rt"])),
    "BLTUC": _branch(lambda m, f: _r(m, f["rs"]) < _r(m, f["rt"])),
    "BEQIC": _branch(lambda m, f: _r(m, f["rt"]) == f["u"]),
    "BNEIC": _branch(lambda m, f: _r(m, f["rt"]) != f["u"]),
    "BGEIC": _branch(lambda m, f: _s(m, f["rt"]) >= f["u"]),
    "BLTIC": _branch(lambda m, f: _s(m, f["rt"]) < f["u"]),
    "BGEIUC": _branch(lambda m, f: _r(m, f["rt"]) >= f["u"]),
    "BLTIUC": _branch(lambda m, f: _r(m, f["rt"]) < f["u"]),
    "BEQZC": _branch(lambda m, f: _r(m, f["rt"]) == 0),
    "BNEZC": _branch(lambda m, f: _r(m, f["rt"]) != 0),
    "BBEQZC": _bbit(0), "BBNEZC": _bbit(1),
    "BALRSC": _brsc(True), "BRSC": _brsc(False),
    "JALRC": _jalrc, "JALRC.HB": _jalrc_hb, "JRC": _jrc, "MOVE.BALC": _move_balc,
    # loads and stores
    "LB": _ld(1, True), "LBU": _ld(1, False), "LH": _ld(2, True), "LHU": _ld(2, False),
    "LW": _ld(4, True), "SB": _st(1), "SH": _st(2), "SW": _st(4),
    "LBE": _ld(1, True), "LBUE": _ld(1, False), "LHE": _ld(2, True), "LHUE": _ld(2, False),
    "LWE": _ld(4, True), "SBE": _st(1), "SHE": _st(2), "SWE": _st(4),
    "UALH": _ld(2, True), "UASH": _st(2),
    "LBX": _ldx(1, True), "LBUX": _ldx(1, False), "LHX": _ldx(2, True),
    "LHUX": _ldx(2, False), "LWX": _ldx(4, True), "LHXS": _ldx(2, True, 1),
    "LHUXS": _ldx(2, False, 1), "LWXS": _ldx(4, True, 2),
    "SBX": _stx(1), "SHX": _stx(2), "SWX": _stx(4), "SHXS": _stx(2, 1), "SWXS": _stx(4, 2),
    "LWPC": _lwpc, "SWPC": _swpc,
    "LWM": _lwm, "UALWM": _lwm, "SWM": _swm, "UASWM": _swm,
    "SAVE": _save, "RESTORE": _restore, "LL": _ll, "SC": _sc,
    # system
    "MFC0": _mfc0(False), "MFHC0": _mfc0(True), "MTC0": _mtc0(False), "MTHC0": _mtc0(True),
    "DI": lambda m: S._EXECUTORS["DI"](m, _f(m)["rt"] << 16),
    "EI": lambda m: S._EXECUTORS["EI"](m, _f(m)["rt"] << 16),
    "ERET": _eret, "DERET": lambda m: S._EXECUTORS["DERET"](m, 0),
    "WAIT": lambda m: S._EXECUTORS["WAIT"](m, 0),
    "RDPGPR": lambda m: S._EXECUTORS["RDPGPR"](m, _f(m)["rs"] << 16 | _f(m)["rt"] << 11),
    "WRPGPR": lambda m: S._EXECUTORS["WRPGPR"](m, _f(m)["rs"] << 16 | _f(m)["rt"] << 11),
    "TLBP": _tlb("tlbp"), "TLBR": _tlb("tlbr"), "TLBWI": _tlb("tlbwi"),
    "TLBWR": _tlb("tlbwr"), "TLBINV": _tlb("tlbinv", True),
    "TLBINVF": _tlb("tlbinvf", True), "GINVI": _ginv(2), "GINVT": _ginv(3),
    "DVP": _vp("DVP"), "EVP": _vp("EVP"), "RDHWR": _rdhwr,
    "TEQ": _trap(True), "TNE": _trap(False),
    "BREAK": lambda m: S._EXECUTORS["BREAK"](m, (_f(m)["code"] & 0xFFFFF) << 6),
    "SDBBP": lambda m: S._EXECUTORS["SDBBP"](m, (_f(m)["code"] & 0xFFFFF) << 6),
    "SYSCALL": _syscall, "SIGRIE": _sigrie,
    "CACHE": _cache, "PREF": _pref, "SYNCI": _synci,
    "SYNC": lambda m: m.sync(_f(m)["stype"]), "EHB": _nop, "PAUSE": _nop, "NOP": _nop,
}


# ---------------------------------------------------------------- the DSP ASE
#
# MD01249: every page but BPOSGE32C is the MIPS DSP Module's operation
# (MD00374, emips.dsp) on nanoMIPS's fields — so each runs its MIPS32 twin
# on a MIPS32 word holding the same operands: page -> (the twin row, (nanoMIPS
# field, the twin template's letter) per operand, in the assembler's
# operand order).  Generated with the rows from binutils' D32 entries.

_DSP_TWINS: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    'ABSQ_S.PH': ('ABSQ_S.PH', (('rt', 'd'), ('rs', 't'))),
    'ABSQ_S.QB': ('ABSQ_S.QB', (('rt', 'd'), ('rs', 't'))),
    'ABSQ_S.W': ('ABSQ_S.W', (('rt', 'd'), ('rs', 't'))),
    'ADDQ.PH': ('ADDQ.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'ADDQH.PH': ('ADDQH.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'ADDQH.W': ('ADDQH.W', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'ADDQH_R.PH': ('ADDQH_R.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'ADDQH_R.W': ('ADDQH_R.W', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'ADDQ_S.PH': ('ADDQ_S.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'ADDQ_S.W': ('ADDQ_S.W', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'ADDSC': ('ADDSC', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'ADDU.PH': ('ADDU.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'ADDU.QB': ('ADDU.QB', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'ADDUH.QB': ('ADDUH.QB', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'ADDUH_R.QB': ('ADDUH_R.QB', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'ADDU_S.PH': ('ADDU_S.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'ADDU_S.QB': ('ADDU_S.QB', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'ADDWC': ('ADDWC', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'APPEND': ('APPEND', (('rt', 't'), ('rs', 's'), ('sa', 'a'))),
    'BITREV': ('BITREV', (('rt', 'd'), ('rs', 't'))),
    'CMP.EQ.PH': ('CMP.EQ.PH', (('rs', 's'), ('rt', 't'))),
    'CMP.LE.PH': ('CMP.LE.PH', (('rs', 's'), ('rt', 't'))),
    'CMP.LT.PH': ('CMP.LT.PH', (('rs', 's'), ('rt', 't'))),
    'CMPGDU.EQ.QB': ('CMPGDU.EQ.QB', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'CMPGDU.LE.QB': ('CMPGDU.LE.QB', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'CMPGDU.LT.QB': ('CMPGDU.LT.QB', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'CMPGU.EQ.QB': ('CMPGU.EQ.QB', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'CMPGU.LE.QB': ('CMPGU.LE.QB', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'CMPGU.LT.QB': ('CMPGU.LT.QB', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'CMPU.EQ.QB': ('CMPU.EQ.QB', (('rs', 's'), ('rt', 't'))),
    'CMPU.LE.QB': ('CMPU.LE.QB', (('rs', 's'), ('rt', 't'))),
    'CMPU.LT.QB': ('CMPU.LT.QB', (('rs', 's'), ('rt', 't'))),
    'DPA.W.PH': ('DPA.W.PH', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'DPAQX_S.W.PH': ('DPAQX_S.W.PH', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'DPAQX_SA.W.PH': ('DPAQX_SA.W.PH', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'DPAQ_S.W.PH': ('DPAQ_S.W.PH', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'DPAQ_SA.L.W': ('DPAQ_SA.L.W', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'DPAU.H.QBL': ('DPAU.H.QBL', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'DPAU.H.QBR': ('DPAU.H.QBR', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'DPAX.W.PH': ('DPAX.W.PH', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'DPS.W.PH': ('DPS.W.PH', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'DPSQX_S.W.PH': ('DPSQX_S.W.PH', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'DPSQX_SA.W.PH': ('DPSQX_SA.W.PH', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'DPSQ_S.W.PH': ('DPSQ_S.W.PH', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'DPSQ_SA.L.W': ('DPSQ_SA.L.W', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'DPSU.H.QBL': ('DPSU.H.QBL', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'DPSU.H.QBR': ('DPSU.H.QBR', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'DPSX.W.PH': ('DPSX.W.PH', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'EXTP': ('EXTP', (('rt', 't'), ('ac', 'A'), ('size', 'Z'))),
    'EXTPDP': ('EXTPDP', (('rt', 't'), ('ac', 'A'), ('size', 'Z'))),
    'EXTPDPV': ('EXTPDPV', (('rt', 't'), ('ac', 'A'), ('rs', 's'))),
    'EXTPV': ('EXTPV', (('rt', 't'), ('ac', 'A'), ('rs', 's'))),
    'EXTR.W': ('EXTR.W', (('rt', 't'), ('ac', 'A'), ('sa', 'f'))),
    'EXTRV.W': ('EXTRV.W', (('rt', 't'), ('ac', 'A'), ('rs', 's'))),
    'EXTRV_R.W': ('EXTRV_R.W', (('rt', 't'), ('ac', 'A'), ('rs', 's'))),
    'EXTRV_RS.W': ('EXTRV_RS.W', (('rt', 't'), ('ac', 'A'), ('rs', 's'))),
    'EXTRV_S.H': ('EXTRV_S.H', (('rt', 't'), ('ac', 'A'), ('rs', 's'))),
    'EXTR_R.W': ('EXTR_R.W', (('rt', 't'), ('ac', 'A'), ('sa', 'f'))),
    'EXTR_RS.W': ('EXTR_RS.W', (('rt', 't'), ('ac', 'A'), ('sa', 'f'))),
    'EXTR_S.H': ('EXTR_S.H', (('rt', 't'), ('ac', 'A'), ('sa', 'f'))),
    'INSV': ('INSV', (('rt', 't'), ('rs', 's'))),
    'MADD': ('MADD@DSP', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'MADDU': ('MADDU@DSP', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'MAQ_S.W.PHL': ('MAQ_S.W.PHL', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'MAQ_S.W.PHR': ('MAQ_S.W.PHR', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'MAQ_SA.W.PHL': ('MAQ_SA.W.PHL', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'MAQ_SA.W.PHR': ('MAQ_SA.W.PHR', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'MFHI': ('MFHI@DSP', (('rt', 'd'), ('ac', 'A'))),
    'MFLO': ('MFLO@DSP', (('rt', 'd'), ('ac', 'A'))),
    'MODSUB': ('MODSUB', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'MSUB': ('MSUB@DSP', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'MSUBU': ('MSUBU@DSP', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'MTHI': ('MTHI@DSP', (('rs', 's'), ('ac', 'A'))),
    'MTHLIP': ('MTHLIP', (('rs', 's'), ('ac', 'A'))),
    'MTLO': ('MTLO@DSP', (('rs', 's'), ('ac', 'A'))),
    'MUL.PH': ('MUL.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'MULEQ_S.W.PHL': ('MULEQ_S.W.PHL', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'MULEQ_S.W.PHR': ('MULEQ_S.W.PHR', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'MULEU_S.PH.QBL': ('MULEU_S.PH.QBL', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'MULEU_S.PH.QBR': ('MULEU_S.PH.QBR', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'MULQ_RS.PH': ('MULQ_RS.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'MULQ_RS.W': ('MULQ_RS.W', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'MULQ_S.PH': ('MULQ_S.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'MULQ_S.W': ('MULQ_S.W', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'MULSA.W.PH': ('MULSA.W.PH', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'MULSAQ_S.W.PH': ('MULSAQ_S.W.PH', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'MULT': ('MULT@DSP', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'MULTU': ('MULTU@DSP', (('ac', 'A'), ('rs', 's'), ('rt', 't'))),
    'MUL_S.PH': ('MUL_S.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'PACKRL.PH': ('PACKRL.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'PICK.PH': ('PICK.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'PICK.QB': ('PICK.QB', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'PRECEQ.W.PHL': ('PRECEQ.W.PHL', (('rt', 'd'), ('rs', 't'))),
    'PRECEQ.W.PHR': ('PRECEQ.W.PHR', (('rt', 'd'), ('rs', 't'))),
    'PRECEQU.PH.QBL': ('PRECEQU.PH.QBL', (('rt', 'd'), ('rs', 't'))),
    'PRECEQU.PH.QBLA': ('PRECEQU.PH.QBLA', (('rt', 'd'), ('rs', 't'))),
    'PRECEQU.PH.QBR': ('PRECEQU.PH.QBR', (('rt', 'd'), ('rs', 't'))),
    'PRECEQU.PH.QBRA': ('PRECEQU.PH.QBRA', (('rt', 'd'), ('rs', 't'))),
    'PRECEU.PH.QBL': ('PRECEU.PH.QBL', (('rt', 'd'), ('rs', 't'))),
    'PRECEU.PH.QBLA': ('PRECEU.PH.QBLA', (('rt', 'd'), ('rs', 't'))),
    'PRECEU.PH.QBR': ('PRECEU.PH.QBR', (('rt', 'd'), ('rs', 't'))),
    'PRECEU.PH.QBRA': ('PRECEU.PH.QBRA', (('rt', 'd'), ('rs', 't'))),
    'PRECR.QB.PH': ('PRECR.QB.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'PRECRQ.PH.W': ('PRECRQ.PH.W', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'PRECRQ.QB.PH': ('PRECRQ.QB.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'PRECRQU_S.QB.PH': ('PRECRQU_S.QB.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'PRECRQ_RS.PH.W': ('PRECRQ_RS.PH.W', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'PRECR_SRA.PH.W': ('PRECR_SRA.PH.W', (('rt', 't'), ('rs', 's'), ('sa', 'a'))),
    'PRECR_SRA_R.PH.W': ('PRECR_SRA_R.PH.W', (('rt', 't'), ('rs', 's'), ('sa', 'a'))),
    'RADDU.W.QB': ('RADDU.W.QB', (('rt', 'd'), ('rs', 's'))),
    'RDDSP': ('RDDSP', (('rt', 'd'), ('mask', 'm'))),
    'REPL.PH': ('REPL.PH', (('rt', 'd'), ('imm', 'i'))),
    'REPL.QB': ('REPL.QB', (('rt', 'd'), ('imm', 'i'))),
    'REPLV.PH': ('REPLV.PH', (('rt', 'd'), ('rs', 't'))),
    'REPLV.QB': ('REPLV.QB', (('rt', 'd'), ('rs', 't'))),
    'SHILO': ('SHILO', (('ac', 'A'), ('shift', 'f'))),
    'SHILOV': ('SHILOV', (('ac', 'A'), ('rs', 's'))),
    'SHLL.PH': ('SHLL.PH', (('rt', 'd'), ('rs', 't'), ('sa', 'f'))),
    'SHLL.QB': ('SHLL.QB', (('rt', 'd'), ('rs', 't'), ('sa', 'f'))),
    'SHLLV.PH': ('SHLLV.PH', (('rd', 'd'), ('rt', 't'), ('rs', 's'))),
    'SHLLV.QB': ('SHLLV.QB', (('rd', 'd'), ('rt', 't'), ('rs', 's'))),
    'SHLLV_S.PH': ('SHLLV_S.PH', (('rd', 'd'), ('rt', 't'), ('rs', 's'))),
    'SHLLV_S.W': ('SHLLV_S.W', (('rd', 'd'), ('rt', 't'), ('rs', 's'))),
    'SHLL_S.PH': ('SHLL_S.PH', (('rt', 'd'), ('rs', 't'), ('sa', 'f'))),
    'SHLL_S.W': ('SHLL_S.W', (('rt', 'd'), ('rs', 't'), ('sa', 'f'))),
    'SHRA.PH': ('SHRA.PH', (('rt', 'd'), ('rs', 't'), ('sa', 'f'))),
    'SHRA.QB': ('SHRA.QB', (('rt', 'd'), ('rs', 't'), ('sa', 'f'))),
    'SHRAV.PH': ('SHRAV.PH', (('rd', 'd'), ('rt', 't'), ('rs', 's'))),
    'SHRAV.QB': ('SHRAV.QB', (('rd', 'd'), ('rt', 't'), ('rs', 's'))),
    'SHRAV_R.PH': ('SHRAV_R.PH', (('rd', 'd'), ('rt', 't'), ('rs', 's'))),
    'SHRAV_R.QB': ('SHRAV_R.QB', (('rd', 'd'), ('rt', 't'), ('rs', 's'))),
    'SHRAV_R.W': ('SHRAV_R.W', (('rd', 'd'), ('rt', 't'), ('rs', 's'))),
    'SHRA_R.PH': ('SHRA_R.PH', (('rt', 'd'), ('rs', 't'), ('sa', 'f'))),
    'SHRA_R.QB': ('SHRA_R.QB', (('rt', 'd'), ('rs', 't'), ('sa', 'f'))),
    'SHRA_R.W': ('SHRA_R.W', (('rt', 'd'), ('rs', 't'), ('sa', 'f'))),
    'SHRL.PH': ('SHRL.PH', (('rt', 'd'), ('rs', 't'), ('sa', 'f'))),
    'SHRL.QB': ('SHRL.QB', (('rt', 'd'), ('rs', 't'), ('sa', 'f'))),
    'SHRLV.PH': ('SHRLV.PH', (('rd', 'd'), ('rt', 't'), ('rs', 's'))),
    'SHRLV.QB': ('SHRLV.QB', (('rd', 'd'), ('rt', 't'), ('rs', 's'))),
    'SUBQ.PH': ('SUBQ.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'SUBQH.PH': ('SUBQH.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'SUBQH.W': ('SUBQH.W', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'SUBQH_R.PH': ('SUBQH_R.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'SUBQH_R.W': ('SUBQH_R.W', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'SUBQ_S.PH': ('SUBQ_S.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'SUBQ_S.W': ('SUBQ_S.W', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'SUBU.PH': ('SUBU.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'SUBU.QB': ('SUBU.QB', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'SUBUH.QB': ('SUBUH.QB', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'SUBUH_R.QB': ('SUBUH_R.QB', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'SUBU_S.PH': ('SUBU_S.PH', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'SUBU_S.QB': ('SUBU_S.QB', (('rd', 'd'), ('rs', 's'), ('rt', 't'))),
    'WRDSP': ('WRDSP', (('rt', 's'), ('mask', 'm'))),
}


def _twin_layout(twin: str, pairs: tuple) -> tuple[int, tuple]:
    """The twin's fixed bits and (field, shift, width) of each operand."""
    tpl = ISA.for_release("R6", dsp=2).get(twin)["encode"]
    fixed = int("".join(c if c in "01" else "0" for c in tpl), 2)
    out = []
    for name, letter in pairs:
        msb, lsb = tpl.index(letter), tpl.rindex(letter)
        out.append((name, 31 - lsb, lsb - msb + 1))
    return fixed, tuple(out)


_LAYOUTS: dict[str, tuple[int, tuple]] = {}


def twin_word(page: str, f: dict) -> int:
    """The MIPS32 twin's word of a DSP ASE instruction with fields ``f``."""
    lay = _LAYOUTS.get(page)
    if lay is None:
        lay = _LAYOUTS[page] = _twin_layout(*_DSP_TWINS[page])
    w = lay[0]
    for name, shift, width in lay[1]:
        w |= (f[name] & ((1 << width) - 1)) << shift
    return w


#: twin words -> their Decode (views and the pipeline)
_TWINS: dict[int, Decode] = {}


def twin(i: NM.NM) -> Decode | None:
    """The MIPS32 twin (a Decode of the MIPS DSP Module's row) of a DSP ASE
    instruction, None for any other."""
    page = i.row.name
    if page not in _DSP_TWINS:
        return None
    w = twin_word(page, i.f)
    dec = _TWINS.get(w)
    if dec is None:
        if len(_TWINS) > 65536:
            _TWINS.clear()
        dec = _TWINS[w] = ISA.for_release("R6", dsp=2).decode(w)
    return dec


def _twin_gprs(page: str, f: dict, written: bool) -> tuple[int, ...]:
    from .pipeline import analyze
    w = twin_word(page, f)
    dec = _TWINS.get(w) or ISA.for_release("R6", dsp=2).decode(w)
    info = analyze(dec)
    regs = info.writes if written else info.reads_id | info.reads_ex
    return tuple(sorted(r[1] for r in regs if isinstance(r, tuple) and r[0] == "r"))


def _dsp_page(page: str) -> Callable[[Any], Any]:
    run = S._EXECUTORS[_DSP_TWINS[page][0]]
    return lambda m: run(m, twin_word(page, m._nm.f))


def _bposge32c(m: Any) -> None:
    """BPOSGE32C (MD01249): taken when DSPControl.pos >= 32; the target is
    the next instruction + offset, as every nanoMIPS branch's (the page's
    Operation text is MIPS32's: DEVIATIONS)."""
    if m.dspctl & 0x3F >= 32:
        m._redirect = (_next(m) + _f(m)["offset"]) & M32


_PAGES.update({page: _dsp_page(page) for page in _DSP_TWINS})


# ---------------------------------------------------------------- the MT ASE
#
# MD01255: the MIPS MT Module (MD00378, emips.mt) on nanoMIPS's fields —
# DMT / EMT / DVPE / EVPE / FORK / YIELD run emips.mt's executors on a
# MIPS32 word with the same registers; MFTR / MTTR call the processor with
# their 5-bit sel (MIPS32's has 3 bits and rx; nanoMIPS's has no rx).

def _mt_page(name: str, word: Callable[[dict], int]) -> Callable[[Any], Any]:
    from . import mt as MT
    run = MT.EXECUTORS[name]
    return lambda m: run(m, word(_f(m)))


def _mftr(m: Any) -> None:
    f, p = _f(m), m.mt
    tc = p.target(m)
    value = m.amask if tc is None else p.mftr(m, tc, f["rs"], f["u"], f["sel"], f["h"], 0)
    m.write_reg(f["rt"], value)


def _mttr(m: Any) -> None:
    f, p = _f(m), m.mt
    tc = p.target(m)
    if tc is not None:
        p.mttr(m, tc, f["rs"], f["u"], f["sel"], f["h"], 0, m.read_reg(f["rt"]))


_PAGES.update({
    "DMT": _mt_page("DMT", lambda f: f["rt"] << 16),
    "EMT": _mt_page("EMT", lambda f: f["rt"] << 16),
    "DVPE": _mt_page("DVPE", lambda f: f["rt"] << 16),
    "EVPE": _mt_page("EVPE", lambda f: f["rt"] << 16),
    "FORK": _mt_page("FORK", lambda f: f["rs"] << 21 | f["rt"] << 16 | f["rd"] << 11),
    "YIELD": _mt_page("YIELD", lambda f: f["rs"] << 21 | f["rt"] << 11),
    "MFTR": _mftr, "MTTR": _mttr,
})
_PAGES["BPOSGE32C"] = _bposge32c
assert set(_PAGES) == NM.NAMES, set(_PAGES) ^ NM.NAMES

#: the executors (into sim._EXECUTORS): one per entry
EXECUTORS: dict[str, S.Executor] = {
    name: _wrap(_PAGES[name[3:]] if name[3:] in _PAGES else
                _PAGES[NM.TABLE.by_form[name[3:]].name])
    for name in ENTRIES}


# ---------------------------------------------------------------- what a page uses
#
# The GPRs each instruction page reads and writes, by the names its decode
# steps give them (emips.nanomips: the Operation's variables).  The
# pipeline's hazards (PLAN M23 phase 4b) and the views' operands use them;
# tests/test_nanomips_session.py checks them against the executors.

def _keys(*names: str) -> Callable[[dict], tuple[int, ...]]:
    return lambda f: tuple(f[n] for n in names)


#: page -> (reads, writes): functions of the decoded fields giving GPR numbers
_USES: dict[str, tuple[Callable[[dict], tuple], Callable[[dict], tuple]]] = {}


def _use(pages: str, reads: Callable[[dict], tuple], writes: Callable[[dict], tuple]) -> None:
    for page in pages.split():
        _USES[page] = (reads, writes)


_NONE: Callable[[dict], tuple] = lambda f: ()            # noqa: E731
_use("ADD SUB SOV SUBU AND OR XOR NOR SLT SLTU SLLV SRLV SRAV ROTRV MUH MULU MUHU DIV DIVU "
     "MOD MODU", _keys("rs", "rt"), _keys("rd"))
_use("ADDU MUL", _keys("src1", "src2"), _keys("dst"))
_use("MOVN MOVZ", _keys("rs", "rt", "rd"), _keys("rd"))        # (rd kept when not moved)
_use("NOT MOVE CLO CLZ SEB SEH EXT ROTX ADDIU ANDI ORI XORI SLTI SLTIU SEQI SLL SRL SRA ROTR",
     _keys("rs"), _keys("rt"))
_use("INS CRC32B CRC32H CRC32W CRC32CB CRC32CH CRC32CW", _keys("rs", "rt"), _keys("rt"))
_use("EXTW LSA", _keys("rs", "rt"), _keys("rd"))
_use("MOVEP", _keys("src1", "src2"), _keys("dst1", "dst2"))
_use("ADDIUPC ALUIPC LI LUI MFC0 MFHC0 DI EI DVP EVP RDHWR", _NONE, _keys("rt"))
# (the previous shadow set's rs: with one set, the current one's)
_use("RDPGPR", _keys("rs"), _keys("rt"))
_use("BC ERET DERET WAIT TLBP TLBR TLBWI TLBWR TLBINV TLBINVF BREAK SDBBP SYSCALL SIGRIE "
     "SYNC EHB PAUSE NOP", _NONE, _NONE)
_use("BALC", _NONE, lambda f: (31,))
_use("BEQC BNEC BGEC BLTC BGEUC BLTUC TEQ TNE", _keys("rs", "rt"), _NONE)
_use("BEQIC BNEIC BGEIC BLTIC BGEIUC BLTIUC BEQZC BNEZC BBEQZC BBNEZC JRC MTC0 MTHC0 SWPC",
     _keys("rt"), _NONE)
_use("BALRSC JALRC.HB", _keys("rs"), _keys("rt"))
_use("BRSC CACHE PREF SYNCI GINVI GINVT", _keys("rs"), _NONE)
_use("JALRC", _keys("src"), _keys("dst"))
_use("MOVE.BALC", _keys("rt"), lambda f: (f["rd"], 31))
_use("LB LBU LH LHU LW LBE LBUE LHE LHUE LWE UALH", _keys("rs"), _keys("rt"))
_use("SB SH SW SBE SHE SWE UASH", _keys("rs", "rt"), _NONE)
_use("LBX LBUX LHX LHUX LWX LHXS LHUXS LWXS", _keys("rs", "rt"), _keys("rd"))
_use("SBX SHX SWX SHXS SWXS", _keys("rs", "rt", "rd"), _NONE)
_use("LWPC", _NONE, _keys("rt"))
_use("LWM UALWM", _keys("rs"), lambda f: tuple(_list(f["rt"], f["count"])))
_use("SWM UASWM", lambda f: (f["rs"],) + (() if f["rt"] == 0 else
                                         tuple(_list(f["rt"], f["count"]))), _NONE)
_use("SAVE", lambda f: (29,) + tuple(_list(f["rt"], f["count"], f["gp"])), lambda f: (29,))
_use("RESTORE", lambda f: (29,), lambda f: tuple(_list(f["rt"], f["count"], f["gp"])) + (29,))
_use("LL", lambda f: (f["rs"],), lambda f: (f["rt"], f["ru"]) if f["nbytes"] == 8
     else (f["rt"],))
_use("SC", lambda f: (f["rs"], f["rt"]) + ((f["ru"],) if f["nbytes"] == 8 else ()),
     _keys("rt"))
# (WRPGPR's Operation: SRS[PSS][rt] = GPR[rs] — the ch. 2 summary's
# "rs = SRS[PSS][rt]" has it the wrong way round)
_use("WRPGPR", _keys("rs"), _keys("rt"))
# the DSP ASE: its MIPS32 twin's registers (emips.pipeline.analyze, whose
# DSP tables tests/test_dsp_pipeline.py checks against the executors;
# binutils' RD_n / WR_n flags are wrong for some: MTHLIP "writes" rs)
for _page in _DSP_TWINS:
    _USES[_page] = (lambda f, p=_page: _twin_gprs(p, f, False),
                    lambda f, p=_page: _twin_gprs(p, f, True))
_use("BPOSGE32C", _NONE, _NONE)
# the MT ASE (FORK writes the new TC's register, MTTR the target TC's)
_use("DMT EMT DVPE EVPE MFTR", _NONE, _keys("rt"))
_use("FORK", _keys("rs", "rt"), _NONE)
_use("YIELD", _keys("rs"), _keys("rt"))
_use("MTTR", _keys("rt"), _NONE)
assert set(_USES) == NM.NAMES, set(_USES) ^ NM.NAMES


def uses(i: NM.NM) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(the GPRs the instruction reads, the GPRs it writes) — $0 left out;
    a register list (SAVE, LWM ...) in its order."""
    reads, writes = _USES[i.row.name]
    f = i.f
    return (tuple(r for r in reads(f) if r)), tuple(r for r in writes(f) if r)


def link(i: NM.NM, pc: int) -> int | None:
    """The return address of a call at ``pc`` (BALC, MOVE.BALC, JALRC /
    JALRC.HB / BALRSC with a link register): the next instruction; None
    when ``i`` is not a call."""
    name, f = i.row.name, i.f
    call = (name in ("BALC", "MOVE.BALC") or name == "JALRC" and f["dst"] != 0
            or name in ("JALRC.HB", "BALRSC") and f["rt"] != 0)
    return (pc + i.size) & M32 if call else None


def returns(i: NM.NM) -> bool:
    """Is ``i`` a return: a jump to $31 that links nothing (``jrc $ra``,
    ``jalrc $zero, $ra``, ``jalrc.hb $zero, $ra``) or RESTORE.JRC (to the
    restored $31)?"""
    name, f = i.row.name, i.f
    return (name == "JRC" and f["rt"] == 31
            or name == "JALRC" and f["dst"] == 0 and f["src"] == 31
            or name == "JALRC.HB" and f["rt"] == 0 and f["rs"] == 31
            or name == "RESTORE" and bool(f["jr"]))
