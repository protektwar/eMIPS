"""The MIPS64 machine (PLAN M22 phase 2): 64-bit registers and the
doubleword instructions.

Sources: *MIPS64 Architecture for Programmers* Vol. I (MD00083 rev.
6.01: the 64-bit programming model, 32-bit compatibility), Vol. II
(MD00087 rev. 5.04 / 6.06: every instruction's MIPS64 operation) and
Vol. III (MD00091 rev. 5.04 / 6.03: 3.5 the 64-bit enables, 9.30 Status,
9.43 Config.AT).

``Machine(arch="mips64")`` builds a :class:`Machine64`.  Its GPRs, HI /
LO, PC and the address registers of CP0 (EPC, ErrorEPC, DEPC, DESAVE,
BadVAddr, UserLocal) are 64 bits wide.  Two address configurations
(Vol. III Table 9.44, Config.AT):

* **AT = 1** with the fixed mapping MMU — "MIPS64 with access only to
  32-bit compatibility segments"; App. A.1: a fixed-mapping MIPS64 core
  "is assumed to have 32-bit addresses".  A virtual address must be a
  sign-extended 32-bit value — useg / kseg0 / kseg1 / sseg / kseg3 at
  ``0x0000_0000_xxxx_xxxx`` and ``0xFFFF_FFFF_8xxx_xxxx`` up — and any
  other 64-bit address is an Address Error, as on every MIPS64 core while
  Status.KX / SX / UX are 0; those three bits "must be ignored on write
  and read as zero" (9.30).
* **AT = 2** with a TLB (PLAN M22 phase 3) — the whole 64-bit address
  space of Table 4.1 / 4.2: xuseg (UX), xsseg (SX), xkphys and xkseg
  (KX), the compatibility segments; SEGBITS = 40, PABITS = 32
  (:data:`emips.tlb.SEGBITS`); Status.KX / SX / UX writable (SX with
  supervisor mode); a TLB Refill of a reference with 64-bit addresses
  enabled goes to the XTLB Refill vector (offset 0x080, Table 6.9);
  EntryHi (R, VPN2 39:13), Context and XContext are 64 bits, EntryLo has
  RI / XI at 63 / 62 for DMFC0 / DMTC0 (MFC0 / MTC0 see them at 31 / 30,
  Vol. II MFC0 / MTC0).

With the BAT (App. A.2: base-and-bounds entries for useg, kseg2 and
kseg3 only) the core has 32-bit addresses too (AT = 1).  With a TLB the
FTLB, Segmentation Control (+ the xkphys regions of SegCtl1.XAM /
SegCtl2.XR, 4.12; EBase.WG makes EBase 64 bits) and the page table walker
(64-bit pointers with PWSize.PS, PWCtl.XK / XS / XU, 4.14) work as the
P6600's (MD01138).

Status.PX (23) is writable and enables the 64-bit operations in User Mode
(3.5.2).  32-bit compatibility addressing: before Release 6, in User Mode
with UX = 0 the effective address of a load, store or prefetch is
sign-extended from bit 31 before it is checked (5.04 4.10); Release 6 and
a Release 5 core with EVA (6.03 4.11, Config5.EVA) do it in every mode
whose 64-bit address enable (KX / SX / UX) is 0 — an EVA instruction's by
User Mode's —, for data references (Load / Store / CACHE / PREF / LL / SC
/ SYNCI: a base register that is not a sign-extended word is an Address
Error, before Release 6 an indexed one's index register too) and for
instruction addresses (a branch, jump or the next PC that carries out of
bit 31 wraps; a jump register or ERET to an address that is not
sign-extended faults at the target), and writing KX = 0 clears SX, KX or
SX = 0 clears UX.

* **32-bit instructions** keep their MIPS32 meaning on the low word and
  sign-extend the result to 64 bits (``Machine64.write_reg``); the ones
  whose result is the full register (AND / OR / XOR / NOR, the logical
  immediates, MOVZ / MOVN / MOVF / MOVT, MFHI / MFLO, the links, SELEQZ
  / SELNEZ, the PC-relative ADDIUPC / AUIPC / ALUIPC) write all 64 bits
  (``write_dreg``); compares, branches and traps compare full registers.
  An operand that is not a sign-extended word ("NotWordValue") makes
  ADD / ADDU / SUB / ... / SRL / ROTR / CLZ / EXT / INS / SEB / WSBH /
  MULT / DIV UNPREDICTABLE: they use its low word, as QEMU does
  (``sim.DEVIATIONS["MIPS64 NotWordValue"]``).
* **64-bit operations** (the instructions the opcode tables mark ⊥, the
  COP1 L / PS formats, COP1X) raise Reserved Instruction in User Mode
  unless Status.PX = 1 (Vol. III 6.2.21; UX is 0 here).
* With AT = 1 the memory system below :meth:`Machine64.data_address`
  and the fetch sees the 32-bit compatibility address (``va & M32``):
  the MMU, caches and devices are those of the MIPS32 core.  With AT = 2
  the TLB sees the 64-bit address (:attr:`Machine.vamask`); physical
  addresses stay 32 bits.  Views take an address of up to 32 bits as a
  compatibility address (``0x80000180`` = kseg0).
"""

from __future__ import annotations

from typing import Any

from . import cp0 as C
from . import sim as S
from . import micromips as MM
from . import mips16 as M16
from .isa import ISA

M32, M64 = S.M32, S.M64
_HIGH = 0xFFFFFFFF00000000
#: Status.PX: 64-bit operations in User Mode (Vol. III 9.30)
ST_PX = 1 << 23
#: Status.KX / SX / UX: 64-bit addressing — read as 0 without it (AT = 1)
ST_KX, ST_SX, ST_UX = 1 << 7, 1 << 6, 1 << 5
#: Config.AT (bits 14:13): 1 = MIPS64, 32-bit compatibility segments only;
#: 2 = MIPS64, all the segments
CONFIG_AT_SHIFT = 13
#: the 64-bit CP0 registers of every MIPS64 core (Vol. III ch. 9)
WIDE_CP0 = (C.EPC, C.ERROREPC, C.DEPC, C.DESAVE, C.BADVADDR, C.USERLOCAL)
#: ... and of a TLB with 64-bit addresses: EntryHi, Context, XContext
WIDE_TLB_CP0 = ((10, 0), (4, 0), (20, 0))
#: EntryLo0 / EntryLo1: RI / XI at 63 / 62 for DMFC0 / DMTC0 (Vol. III 9.6)
ENTRYLO = ((2, 0), (3, 0))
#: ... Segmentation Control's SegCtl0-2 (XAM 61:59, XR 63:56) and EBase
#: (63:12 with the write gate WG), the page table walker's PWBase /
#: PWField / PWSize (Vol. III 9.13-9.18, 9.39)
WIDE_SC_CP0 = ((5, 2), (5, 3), (5, 4), C.EBASE)
WIDE_PW_CP0 = ((5, 5), (5, 6), (5, 7))

_rs, _rt, _rd, _sa = S.rs, S.rt, S.rd, S.sa


def sx32(value: int) -> int:
    """The low word of ``value`` sign-extended to 64 bits (unsigned)."""
    value &= M32
    return value | _HIGH if value & 0x80000000 else value


def s64(value: int) -> int:
    """``value`` as a signed 64-bit number."""
    value &= M64
    return value - (1 << 64) if value >> 63 else value


def compat(addr: int) -> bool:
    """A 64-bit virtual address in the 32-bit compatibility segments: a
    sign-extended 32-bit value (Vol. III 4.3)."""
    return addr < 0x80000000 or addr >= 0xFFFFFFFF80000000


def hole(addr: int, segbits: int, pabits: int) -> bool:
    """An address in none of the segments (Vol. III Table 4.2: Address
    Error in every mode): above xuseg / xsseg / xkseg, or an xkphys
    address with VA 58:PABITS not zero (4.7)."""
    addr &= M64
    if compat(addr):
        return False
    top = addr >> 62
    if top == 2:
        return bool((addr >> pabits) & ((1 << (59 - pabits)) - 1))
    span = (1 << segbits) - (1 << 31 if top == 3 else 0)
    return addr - (top << 62) >= span


def word_value(value: int) -> bool:
    """Vol. II's ``NotWordValue`` is false: a sign-extended word."""
    return sx32(value) == value & M64


def xkphys_ok(m: Any, am: int) -> bool:
    """Segmentation Control (Vol. III 4.12 xkphysRegionLookup): an
    xkphys region enabled in SegCtl2.XR uses SegCtl1.XAM — allowed at the
    access's privilege level (checkAM) and with the 64-bit enable of the
    mode XAM needs at least (KERNEL: UK / MK / 6 = XKP → KX,
    SUPERVISOR: MSK / USK → SX, USER: MUSK / MUSUK / UUSK → UX)."""
    from . import segctl
    if not segctl.allowed(am, segctl.level(m)):
        return False
    need = ST_KX if am in (segctl.UK, segctl.MK, 6) else \
        ST_SX if am in (segctl.MSK, segctl.USK) else ST_UX
    return bool(m.status & need)


def lo64(lo: int) -> int:
    """EntryLo as MIPS64 has it (Vol. III 9.6): RI / XI — kept at 31 / 30,
    where MFC0 sees them — at 63 / 62."""
    return (lo & 0x3FFFFFFF) | ((lo >> 30) & 3) << 62


def _wrapped(addr: int) -> int:
    """Release 6's wrap of an address that carried out of bit 31 (Vol. III
    6.03 4.11: sign-extended from bit 31)."""
    return addr if word_value(addr) else sx32(addr)


def _is_64bit_op(e: dict) -> bool:
    """Needs 64-bit operations enabled (Vol. III 6.2.21): a doubleword
    instruction (⊥ in the opcode tables), a COP1 instruction whose rs is
    L or PS, any COP1X instruction."""
    if e.get("arch") == "mips64":
        return True
    v = e["decode"]["value"]
    return (v >> 26 == 0x11 and (v >> 21) & 31 in (0x15, 0x16)) or v >> 26 == 0x13


class Machine64(S.Machine):
    """A MIPS64 core: Config.AT = 1 (32-bit compatibility addressing) with
    the fixed mapping, 2 (64-bit addresses) with a TLB — see the module
    docstring.  ``Machine(arch="mips64", ...)`` makes one; ``compressed=
    "micromips"`` / ``"micromips-only"`` = microMIPS64 (MD00594, Release
    3+), ``"mips16e"`` = MIPS16e64 (MD00077, Releases 1-5); ``dsp=`` the
    DSP Module (MD00375 rev. 3.02); ``mt=True`` a VPE of the MT Module
    (MD00378 with GPRLEN = 64, Releases 2-5: emips.mt)."""

    arch = "mips64"
    amask = M64
    wide = True
    names = "n64"
    sgn = staticmethod(s64)

    def __init__(self, isa: ISA | None = None, memory: Any = None, pc: int = 0, *,
                 release: str | None = None, compressed: str | None = None,
                 dsp: int | None = None, mt: bool = False, arch: str = "mips64",
                 **kwargs: Any) -> None:
        if isa is None:
            isa = ISA.for_release(release, dsp or 0, mt, arch="mips64")
        if compressed not in (None, "mips16e") and (release or isa.release) in ("R1", "R2"):
            raise ValueError("microMIPS64 needs Release 3+ (MD00594 3.1: microMIPS "
                             "came with Release 3)")
        super().__init__(isa, memory, pc, release=release, compressed=compressed, dsp=dsp,
                         mt=mt, **kwargs)
        #: Config.AT: 2 = 64-bit addresses (a TLB of 64-bit VPN2s:
        #: emips.tlb.TLBMMU.va64), else 1
        self.at = 2 if getattr(self.mmu, "va64", False) else 1
        from .tlb import PABITS, SEGBITS
        self.segbits, self.pabits = (SEGBITS, PABITS) if self.at == 2 else (32, 32)
        if self.at == 2:
            self.vamask = M64                   # the TLB sees the 64-bit address
        else:
            self._mmu_xlate, self._xlate = self._xlate, self._xlate32
        cfg = self.cp0[C.CONFIG]
        cfg["value"] = cfg["value"] & ~(3 << CONFIG_AT_SHIFT) | (self.at << CONFIG_AT_SHIFT)
        mask = self._wmask.get(C.STATUS, M32) & ~(ST_KX | ST_SX | ST_UX)
        #: KX / SX / UX: R/W with 64-bit addressing (SX with supervisor mode)
        self._x64 = ((ST_KX | ST_UX | (ST_SX if getattr(self.mmu, "supervisor", False) else 0))
                     if self.at == 2 else 0)
        self._wmask[C.STATUS] = mask | ST_PX | self._x64
        self.cp0[C.STATUS]["value"] &= ~(ST_KX | ST_SX | ST_UX | ST_PX)
        if self.release == "R1" and self.has_fpu:
            # MIPS64 Release 1 has the 64-bit FPU (FR, the L format: Vol. III
            # 3.5.4), which MIPS32 only gained in Release 2
            self.fir = (S.FIR | (self.prid & 0xFFFF))
            self._wmask[C.STATUS] |= C.ST_FR
        sc, pw = getattr(self.mmu, "sc", False), getattr(self.mmu, "pw", False)
        wide = WIDE_CP0 + tuple(C.WATCHLO) + (WIDE_TLB_CP0 if self.at == 2 else ()) + (
            WIDE_SC_CP0 if sc else ()) + (WIDE_PW_CP0 if pw else ())
        self._wide_cp0 = frozenset(k for k in wide if k in self.cp0)
        self._lo64 = frozenset(ENTRYLO) if self.at == 2 else frozenset()
        for key in self._wide_cp0:
            if self._wmask.get(key, M32) == M32:    # fully writable: all 64 bits
                self._wmask[key] = M64
        for key in C.WATCHLO:                   # WatchLo.VAddr 63:3 (Vol. III 9.50)
            if key in self._wmask:
                self._wmask[key] |= M64 ^ M32
        if sc:                                  # EBase resets to 0xFFFF_FFFF_8000_0000
            self.cp0[C.EBASE]["value"] = sx32(self.cp0[C.EBASE]["value"])
        #: the table MIPS16e64's twins decode with: the core's (Release 1:
        #: Release 2's, for the MIPS16e SEB / SEH / ZEB / ZEH twins)
        self._isa16 = self.isa if self.release != "R1" else ISA.for_release(
            "R2", dsp or 0, arch="mips64")
        #: the rows that need 64-bit operations enabled (ids of ISA entries;
        #: microMIPS64's LDP / LDM / SDP / SDM / LWUPC: MD00594; MIPS16e64's
        #: forms, ⊥ in MD00077 Tables 1.18-1.28: their twins, LDPC /
        #: DADDIUPC / ZEW)
        self._ops64_rows = frozenset(id(e) for e in self.isa.leads if _is_64bit_op(e)) | (
            frozenset(id(e) for e in MM.ENTRIES.values() if e.get("arch") == "mips64")
            if self.compressed == "micromips" else frozenset()) | (
            frozenset(id(e) for e in self._isa16.leads if _is_64bit_op(e)) |
            frozenset(id(e) for e in M16.ENTRIES.values() if e.get("arch") == "mips64")
            if self.compressed == "mips16e" else frozenset())
        #: Release 6's 32-bit compatibility addressing (Vol. III 6.03 4.11;
        #: a Release 5 core too when it has EVA, Config5.EVA): the data
        #: memory references by their base (and index) register fields
        #: (shifts), the jumps whose target is a register (no wrap when it
        #: is not a sign-extended word: the fetch faults) and JIC / JIALC
        #: (register + offset: wraps when the register is a word)
        self.c411 = self.release == "R6" or bool(getattr(self.mmu, "eva", False))
        self._dmem: dict[int, tuple[int, ...]] = {}
        self._reg_jumps: frozenset[int] = frozenset()
        self._jic: frozenset[int] = frozenset()
        self._eva_rows = frozenset(id(e) for e in self.isa.leads if e["name"] in S.EVA_NAMES)
        if self.c411:
            from .syntax import operand_spec
            for e in self.isa.leads:
                try:
                    spec = operand_spec(e)
                except KeyError:
                    continue
                for kind in spec:
                    if kind in ("mem", "mem9", "idx", "mem11"):
                        self._dmem[id(e)] = ((11,) if kind == "mem11" else
                                             (21, 16) if kind == "idx" else (21,))
            self._reg_jumps = frozenset(id(e) for e in self.isa.leads if e["name"] in (
                "JALR", "JALR.HB", "ERET", "DERET"))
            self._jic = frozenset(id(e) for e in self.isa.leads
                                  if e["name"] in ("JIC", "JIALC"))
            if self.compressed:                 # microMIPS64's own forms (raw encodings:
                mm = MM.ENTRIES                 # base 20:16, LWXS's index 25:21; -29 = $sp)
                for form, shifts in (("LWP", (16,)), ("SWP", (16,)), ("LWM32", (16,)),
                                     ("SWM32", (16,)), ("LDP", (16,)), ("SDP", (16,)),
                                     ("LDM", (16,)), ("SDM", (16,)), ("LWXS", (16, 21)),
                                     ("LWM16", (-29,)), ("SWM16", (-29,))):
                    self._dmem[id(mm[form])] = shifts
                self._reg_jumps |= frozenset(id(mm[f]) for f in (
                    "JR16", "JRC", "JALR16", "JALRS16", "JALR", "JALR.HB", "JALRS", "JALRS.HB",
                    "JRADDIUSP", "JRC16", "JALRC16", "JALRC", "JALRC.HB", "JRCADDIUSP"))
                self._jic |= frozenset(id(mm[f]) for f in ("JIC", "JIALC"))   # (rt: 20:16)
        #: this data reference's base register is not a sign-extended word
        #: in a 32-bit compatibility mode (Release 6: an Address Error)
        self._canon_bad = False
        #: this instruction's target may wrap (Release 6, 4.11)
        self._wrap = True

    # -- registers -------------------------------------------------------------

    def read_reg(self, index: int) -> int:
        index &= 31
        return 0 if index == 0 else self.regs[index] & M64

    def write_reg(self, index: int, value: int) -> None:
        """A 32-bit result: its low word, sign-extended (Vol. II's
        ``sign_extend(temp)``)."""
        index &= 31
        if index != 0:
            value &= M32
            self.regs[index] = value | _HIGH if value & 0x80000000 else value

    def write_dreg(self, index: int, value: int) -> None:
        """A 64-bit result."""
        index &= 31
        if index != 0:
            self.regs[index] = value & M64

    def _sxw(self, value: int) -> int:
        return sx32(value)

    def to_pc(self, addr: int) -> int:
        """An address of up to 32 bits is a compatibility address (sign-
        extended); a 64-bit one stays as it is."""
        addr &= M64
        return sx32(addr) if addr <= M32 else addr

    def ops64(self) -> bool:
        """Vol. III 3.5.2: in Kernel, Supervisor or Debug Mode, or with
        Status.PX or UX."""
        status = self.status
        return (self._dm or self.kernel_mode or C.ksu(status) == 1
                or bool(status & (ST_PX | ST_UX)))

    def boot64(self) -> None:
        """What the boot firmware adds on a MIPS64 core, as a 64-bit Linux
        kernel does (arch/mips/kernel/traps.c ``configure_status``):
        Status.FR = 1 (the n64 ABI's 64-bit FPRs) and, with 64-bit
        addresses, KX / SX / UX = 1."""
        bits = (C.ST_FR | self._x64) & self._wmask[C.STATUS]
        self._cp0_set(C.STATUS, self.status | bits)

    def _user32(self) -> bool:
        """User Mode with Status.UX = 0: data addresses are sign-extended
        from bit 31 (Vol. III 4.10)."""
        status = self.status
        return (not self._dm and not C.kernel_mode(status) and C.ksu(status) != 1
                and not status & ST_UX)

    def cp0_view(self, reg: int, sel: int = 0) -> int:
        value = self.cp0_read(reg, sel)
        if (reg, sel) in self._lo64:            # RI / XI at 63 / 62 (DMFC0)
            return lo64(value)
        return value

    def cp0_dread(self, reg: int, sel: int = 0) -> int:
        """DMFC0's value: a 64-bit register whole, a 32-bit one sign-extended
        (as QEMU: the manual does not say, sim.DEVIATIONS); EntryLo with
        64-bit addresses: RI / XI (kept at 31 / 30, MIPS32's place) at 63 /
        62."""
        value = self.cp0_read(reg, sel)
        if (reg, sel) in self._lo64:
            return lo64(value)
        return value if (reg, sel) in self._wide_cp0 else sx32(value)

    def cp0_dwrite(self, reg: int, sel: int, value: int) -> None:
        if (reg, sel) in self._lo64:            # RI / XI from 63 / 62
            value = (value & 0x3FFFFFFF) | ((value >> 62) & 3) << 30
        self.mtc0(reg, sel, value)

    def data_wrap(self) -> bool:
        return self._compat32(self._eva) if self.c411 else self._user32()

    def _compat32(self, eva: bool = False) -> bool:
        """Release 6 / EVA (Vol. III 6.03 4.11): the 64-bit address enable
        of the effective mode is 0 — KX in Kernel / Debug Mode, SX in
        Supervisor Mode, UX in User Mode and for an EVA instruction
        (EffKSU) — so addresses are 32-bit compatible."""
        status = self.status
        if eva:
            return not status & ST_UX
        if self._dm or C.kernel_mode(status):
            return not status & ST_KX
        return not status & (ST_SX if C.ksu(status) == 1 else ST_UX)

    def _bp_va(self, va: int) -> int:
        return sx32(va) if self.at == 1 and va <= M32 else va

    def fetch_va(self, pc: int) -> int:
        return self._compat(pc & M64)

    def seq_pc(self, addr: int) -> int:
        addr &= M64
        return _wrapped(addr) if self.c411 and self._compat32() else addr

    def branch_pc(self, addr: int) -> int:
        addr &= M64
        return _wrapped(addr) if self.c411 and self._wrap and self._compat32() else addr

    def _xlate32(self, m: Any, va: int, kind: str) -> tuple[int, int]:
        """AT = 1: the MMU sees the 32-bit compatibility address; a TLB
        exception (the BAT's) reports it sign-extended in BadVAddr."""
        try:
            return self._mmu_xlate(m, va, kind)
        except S._Arch as exc:
            if exc.badvaddr is not None:
                exc.badvaddr = sx32(exc.badvaddr)
            raise

    def cp0_write(self, reg: int, sel: int = 0, value: int = 0) -> None:
        super().cp0_write(reg, sel, value)
        if (reg, sel) == C.STATUS and self.c411 and self._x64:
            # Release 6 (Vol. III 6.03 9.30): KX = 0 clears SX; KX or SX = 0 UX
            st = self.status
            if not st & ST_KX:
                st &= ~ST_SX
            if not st & ST_KX or (ST_SX & self._x64 and not st & ST_SX):
                st &= ~ST_UX
            self._cp0_set(C.STATUS, st)

    # -- addresses -------------------------------------------------------------

    def _compat(self, addr: int, code: int = C.ADEL) -> int:
        """AT = 1: the 32-bit compatibility address of 64-bit ``addr``, or
        Address Error (``code``) with BadVAddr = ``addr`` — 64-bit
        addressing is not enabled (Vol. III 3.5.1).  AT = 2: ``addr``
        itself (:meth:`address_ok` checks it)."""
        if self.at == 2:
            return addr & M64
        if addr < 0x80000000 or addr >= 0xFFFFFFFF80000000:
            return addr & M32
        raise S._Arch(code, badvaddr=addr)

    def address_ok(self, addr: int) -> bool:
        """AT = 2: Table 4.2 — the compatibility segments as on MIPS32;
        xuseg with UX (every mode), xsseg with SX (Supervisor / Kernel
        Mode), xkphys and xkseg with KX (Kernel Mode); an address in none
        of them never.  Debug Mode has Kernel Mode's rights."""
        addr &= M64
        if self.at == 1 or compat(addr):
            return super().address_ok(addr if self.at == 2 else addr & M32)
        if hole(addr, self.segbits, self.pabits):
            return False
        status = self.status
        top = addr >> 62
        if top == 0:                            # xuseg / xsuseg / xkuseg
            return bool(status & ST_UX)
        if top == 2 and getattr(self.mmu, "sc", False):
            region = (addr >> 59) & 7           # Segmentation Control: SegCtl2.XR
            if self.cp0[(5, 4)]["value"] >> (56 + region) & 1:
                return xkphys_ok(self, (self.cp0[(5, 3)]["value"] >> 59) & 7)
        # (an EVA instruction has User Mode's rights: Vol. III 4.13)
        kernel = not self._eva and (self._dm or C.kernel_mode(status))
        sup = not self._eva and not kernel and C.ksu(status) == 1
        if top == 1:                            # xsseg / xksseg
            return bool(status & ST_SX) and (kernel or sup)
        return kernel and bool(status & ST_KX)  # xkphys, xkseg

    def _ea(self, addr: int) -> int:
        """The virtual address of a data reference: sign-extended from bit
        31 in User Mode with UX = 0 (Vol. III 4.10); Release 6 in every
        32-bit compatibility mode, unless the base register is not a
        sign-extended word (4.11: then an Address Error, :meth:`_canon`)."""
        addr &= M64
        if self.c411:
            return sx32(addr) if not self._canon_bad and self._compat32(self._eva) else addr
        return sx32(addr) if self._user32() else addr

    def _canon(self, addr: int, code: int) -> None:
        """Release 6 (4.11): a data reference whose base register is not
        canonical in a 32-bit compatibility mode is an Address Error at
        base + offset."""
        if self._canon_bad:
            raise S._Arch(code, badvaddr=addr & M64)

    def data_address(self, addr: int, size: int, store: bool, **kwargs: Any) -> int:
        """Machine.data_address of the reference's address (AT = 1: the
        compatibility address); an Address Error reports the 64-bit
        address in BadVAddr."""
        self._canon(addr, C.ADES if store else C.ADEL)
        ea = self._ea(addr)
        va = self._compat(ea, C.ADES if store else C.ADEL)
        span = kwargs.get("span")
        if span is not None:                    # LWL / LWR ...: the same bytes
            kwargs["span"] = ((va + span[0] - addr) & self.vamask, span[1])
        try:
            return super().data_address(va, size, store, **kwargs)
        except S._Arch as exc:
            if self.at == 1 and exc.badvaddr is not None and exc.badvaddr <= M32:
                exc.badvaddr = sx32(exc.badvaddr)
            raise

    def _view(self, addr: int) -> int:
        """A view's address: up to 32 bits a compatibility address (AT = 1
        as is, AT = 2 sign-extended), a 64-bit one when a segment holds
        it; else nothing maps it."""
        if addr <= M32:
            return sx32(addr) if self.at == 2 else addr
        addr &= M64
        if self.at == 2 and not hole(addr, self.segbits, self.pabits):
            return addr
        if compat(addr):
            return addr & M32
        raise S.UnmappedAddress(f"{addr:#018x}: " + (
            "not a 32-bit compatibility address (Config.AT = 1)" if self.at == 1
            else "in no segment (Vol. III Table 4.2)"), addr=addr)

    def _in_dseg(self, addr: int) -> bool:
        """EJTAG's dseg (Vol. III 4.9): 0xFFFF_FFFF_FF20_0000-..._FF3F_FFFF."""
        return (self.at == 1 or addr >> 32 == 0xFFFFFFFF) and super()._in_dseg(addr)

    def translate(self, addr: int) -> Any:
        """MIPS32's, and an address in no segment (or a 64-bit one with
        Config.AT = 1): no physical address, ``fault = "address error"``."""
        try:
            va = self._view(addr)
        except S.UnmappedAddress as exc:
            from .mmu import CCA_UNCACHED, Translation
            return Translation(addr & M64, None, CCA_UNCACHED, "no segment", False,
                               str(exc).split(": ", 1)[1], fault="address error")
        return super().translate(va)

    def physical(self, addr: int) -> int:
        return super().physical(self._view(addr))

    def read_byte(self, addr: int) -> int:
        return super().read_byte(self._view(addr))

    def write_byte(self, addr: int, value: int) -> None:
        super().write_byte(self._view(addr), value)

    def peek_word(self, addr: int) -> int:
        return super().peek_word(self._view(addr))

    def code_word(self, addr: int) -> int:
        return super().code_word(self._view(addr))

    def cache_op(self, op: int, addr: int) -> str:
        """(Release 6: CACHE is a data reference of 4.11; before, 4.10 does
        not name it.)"""
        self._canon(addr, C.ADEL)
        addr = self._ea(addr) if self.c411 else addr & M64
        return super().cache_op(op, self._compat(addr))

    def synci(self, addr: int) -> None:
        self._canon(addr, C.ADEL)
        addr = self._ea(addr) if self.c411 else addr & M64
        super().synci(self._compat(addr))

    def prefetch(self, hint: int, addr: int) -> None:
        if self._canon_bad:                     # PREF: no address errors
            return
        addr = self._ea(addr)                   # (4.10: a prefetch is a data reference)
        if self.at == 2 or compat(addr):
            super().prefetch(hint, addr if self.at == 2 else addr & M32)

    def _log(self, kind: str, addr: int, *rest: Any) -> None:
        """The memory log (views): the 64-bit virtual address."""
        super()._log(kind, sx32(addr) if self.at == 1 else addr & M64, *rest)

    # -- dispatch ---------------------------------------------------------------

    def step(self) -> Any:
        """Release 6 (4.11): in a 32-bit compatibility mode a next PC that
        carries out of bit 31 — the sequential one, a branch or jump target
        — is sign-extended; not the target of a jump register or ERET /
        DERET (a register that is not a sign-extended word faults at the
        target fetch)."""
        if not self.c411 or not self._compat32():
            return super().step()
        outer = self.slot_target if self.slot_armed else None
        self._wrap = True                       # (_dispatch sets it per instruction)
        dec = super().step()
        if self.slot_armed and self.slot_target is not None:     # a branch / jump: its
            if self._wrap:                                       # target, after the slot
                self.slot_target = _wrapped(self.slot_target)
            self.pc = _wrapped(self.pc)                          # the slot: sequential
        elif self._redirect is not None:                         # ERET, compact jumps
            if self._wrap:
                self.pc = _wrapped(self.pc)
        elif outer is None or self.pc != outer & M64:            # sequential
            self.pc = _wrapped(self.pc)
        return dec

    def _dispatch(self, dec: Any, word: int) -> int | None:
        entry = dec.entry
        if self.c411:                           # 4.11: what the inputs are
            self._canon_bad = False
            key = id(entry)
            shifts = self._dmem.get(key)
            self._wrap = True
            if shifts is not None:
                regs = self.regs
                self._canon_bad = (self._compat32(key in self._eva_rows) and not all(
                    word_value(regs[(word >> sh) & 31] if sh >= 0 else regs[-sh])
                    for sh in shifts))
            elif key in self._reg_jumps:
                self._wrap = False
            elif key in self._jic:
                self._wrap = word_value(self.regs[(word >> 16) & 31])
        if id(entry) in self._ops64_rows and not self.ops64():
            # Vol. III 6.2.21: Reserved Instruction "assuming that access to
            # the coprocessor is allowed" — else Coprocessor Unusable first
            fam, name, status = entry["family"], entry["name"], self.status
            if (fam in S.FPU_FAMILIES or name == "PREFX") and not status & C.ST_CU1:
                raise S._Arch(C.CPU, ce=1)
            if fam == "cop2":
                raise S._Arch(C.CPU, ce=2)
            if name in S.PRIVILEGED_NAMES and not (self.kernel_mode or status & C.ST_CU0):
                raise S._Arch(C.CPU, ce=0)
            raise S._Arch(C.RI)
        return super()._dispatch(dec, word)


# ---------------------------------------------------------------------------
# The doubleword instructions (Vol. II-A64 5.04 / 6.06)
# ---------------------------------------------------------------------------

_M = Machine64


def _ov64(r: int) -> int:
    """Integer Overflow when a signed result does not fit in 64 bits."""
    if not -(1 << 63) <= r < (1 << 63):
        raise S._Arch(C.OV)
    return r


def _e_dadd(m: _M, w: int) -> None:
    m.write_dreg(_rd(w), _ov64(s64(m.read_reg(_rs(w))) + s64(m.read_reg(_rt(w)))))


def _e_daddu(m: _M, w: int) -> None:
    m.write_dreg(_rd(w), m.read_reg(_rs(w)) + m.read_reg(_rt(w)))


def _e_dsub(m: _M, w: int) -> None:
    m.write_dreg(_rd(w), _ov64(s64(m.read_reg(_rs(w))) - s64(m.read_reg(_rt(w)))))


def _e_dsubu(m: _M, w: int) -> None:
    m.write_dreg(_rd(w), m.read_reg(_rs(w)) - m.read_reg(_rt(w)))


def _e_daddi(m: _M, w: int) -> None:
    m.write_dreg(_rt(w), _ov64(s64(m.read_reg(_rs(w))) + S._s16(w)))


def _e_daddiu(m: _M, w: int) -> None:
    m.write_dreg(_rt(w), m.read_reg(_rs(w)) + S._s16(w))


def _e_dclo(m: _M, w: int) -> None:
    m.write_dreg(_rd(w), 64 - (~m.read_reg(_rs(w)) & M64).bit_length())


def _e_dclz(m: _M, w: int) -> None:
    m.write_dreg(_rd(w), 64 - m.read_reg(_rs(w)).bit_length())


def _dext(form: str):
    """DEXT / DEXTM / DEXTU rt, rs, pos, size: ``rt ← 0 || rs[pos+size-1
    .. pos]`` (msbd = rd field, lsb = sa field; M: size = msbd + 33, U: pos
    = lsb + 32).  pos + size > 64 is UNPREDICTABLE: the field is cut."""
    def ex(m: _M, w: int) -> None:
        msbd, lsb = _rd(w), _sa(w)
        pos, size = {"": (lsb, msbd + 1), "M": (lsb, msbd + 33),
                     "U": (lsb + 32, msbd + 1)}[form]
        m.write_dreg(_rt(w), (m.read_reg(_rs(w)) >> pos) & ((1 << size) - 1))
    return ex


def _dins(form: str):
    """DINS / DINSM / DINSU rt, rs, pos, size: ``rt[msb..lsb] ←
    rs[msb-lsb..0]`` (M: msb = rd + 32; U: msb = rd + 32, lsb = sa + 32).
    msb < lsb is UNPREDICTABLE: nothing is inserted."""
    def ex(m: _M, w: int) -> None:
        msb, lsb = _rd(w), _sa(w)
        if form:
            msb += 32
        if form == "U":
            lsb += 32
        width = max(msb - lsb + 1, 0)
        mask = (((1 << width) - 1) << lsb) & M64
        src = (m.read_reg(_rs(w)) << lsb) & mask
        m.write_dreg(_rt(w), (m.read_reg(_rt(w)) & ~mask) | src)
    return ex


def _e_dsbh(m: _M, w: int) -> None:
    v = m.read_reg(_rt(w))
    m.write_dreg(_rd(w), ((v & 0x00FF00FF00FF00FF) << 8) | ((v >> 8) & 0x00FF00FF00FF00FF))


def _e_dshd(m: _M, w: int) -> None:
    v = m.read_reg(_rt(w))
    m.write_dreg(_rd(w), sum(((v >> (16 * i)) & 0xFFFF) << (16 * (3 - i)) for i in range(4)))


def _rot64(x: int, n: int) -> int:
    n &= 63
    return ((x >> n) | (x << (64 - n))) & M64 if n else x


def _e_drotr(m: _M, w: int) -> None:
    m.write_dreg(_rd(w), _rot64(m.read_reg(_rt(w)), _sa(w)))


def _e_drotr32(m: _M, w: int) -> None:
    m.write_dreg(_rd(w), _rot64(m.read_reg(_rt(w)), _sa(w) + 32))


def _e_drotrv(m: _M, w: int) -> None:
    m.write_dreg(_rd(w), _rot64(m.read_reg(_rt(w)), m.read_reg(_rs(w))))


def _shift(op: str, extra: int = 0, variable: bool = False):
    """DSLL / DSRL / DSRA (+ 32, variable): 64-bit shifts."""
    def ex(m: _M, w: int) -> None:
        n = (m.read_reg(_rs(w)) & 63) if variable else _sa(w) + extra
        v = m.read_reg(_rt(w))
        m.write_dreg(_rd(w), v << n if op == "l" else v >> n if op == "r" else s64(v) >> n)
    return ex


def _e_dmult(m: _M, w: int) -> None:
    p = s64(m.read_reg(_rs(w))) * s64(m.read_reg(_rt(w)))
    m.hi, m.lo = (p >> 64) & M64, p & M64


def _e_dmultu(m: _M, w: int) -> None:
    p = m.read_reg(_rs(w)) * m.read_reg(_rt(w))
    m.hi, m.lo = p >> 64, p & M64


def _sdiv64(a: int, b: int) -> tuple[int, int]:
    """Quotient toward zero, remainder with the dividend's sign."""
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return q, a - q * b


def _e_ddiv(m: _M, w: int) -> None:
    """DDIV: LO = quotient, HI = remainder.  ÷0 UNPREDICTABLE: all ones
    (as DIV, sim.DEVIATIONS)."""
    a, b = s64(m.read_reg(_rs(w))), s64(m.read_reg(_rt(w)))
    if b == 0:
        m.hi = m.lo = M64
        return
    q, r = _sdiv64(a, b)
    m.lo, m.hi = q & M64, r & M64


def _e_ddivu(m: _M, w: int) -> None:
    a, b = m.read_reg(_rs(w)), m.read_reg(_rt(w))
    if b == 0:
        m.hi = m.lo = M64
    else:
        m.lo, m.hi = a // b, a % b


def _r6_dmuldiv() -> dict[str, S.Executor]:
    """Release 6 DMUL / DMUH / DMULU / DMUHU / DDIV / DMOD / DDIVU /
    DMODU: GPR[rd] ← the 64-bit result (÷0 UNPREDICTABLE: 0, as MUL@R6 &
    co.)."""
    def mk(fn):
        def ex(m: _M, w: int) -> None:
            m.write_dreg(_rd(w), fn(m.read_reg(_rs(w)), m.read_reg(_rt(w))))
        return ex

    def sdiv(a: int, b: int, mod: bool) -> int:
        a, b = s64(a), s64(b)
        if b == 0:
            return 0
        q, r = _sdiv64(a, b)
        return r if mod else q

    return {
        "DMUL": mk(lambda a, b: s64(a) * s64(b)),
        "DMUH": mk(lambda a, b: (s64(a) * s64(b)) >> 64),
        "DMULU": mk(lambda a, b: a * b),
        "DMUHU": mk(lambda a, b: (a * b) >> 64),
        "DDIV@R6": mk(lambda a, b: sdiv(a, b, False)),
        "DMOD": mk(lambda a, b: sdiv(a, b, True)),
        "DDIVU@R6": mk(lambda a, b: a // b if b else 0),
        "DMODU": mk(lambda a, b: a % b if b else 0),
    }


# -- loads / stores --------------------------------------------------------------


def _e_ld(m: _M, w: int) -> None:
    addr = m.data_address(S._addr_i(m, w), 8, store=False)
    m.write_dreg(_rt(w), m.load(addr, 8))


def _e_lwu(m: _M, w: int) -> None:
    addr = m.data_address(S._addr_i(m, w), 4, store=False)
    m.write_dreg(_rt(w), m.load(addr, 4))


def _e_sd(m: _M, w: int) -> None:
    m.store(m.data_address(S._addr_i(m, w), 8, store=True), m.read_reg(_rt(w)), 8)


# LDL / LDR / SDL / SDR (Vol. II-A64 LDL): the doubleword forms of LWL /
# LWR / SWL / SWR — the bytes from the effective address to the end (LDL /
# SDL, big-endian) or start (LDR / SDR) of its aligned doubleword, merged
# into the most / least significant end of the register; the roles of the
# two ends swap in little-endian mode.  ``k = addr & 7``.


def _span_dl(m: _M, addr: int) -> tuple[int, int]:
    """The bytes LDL / SDL reach (the 64-bit EA's; data_address moves
    them with the address it checks)."""
    k = addr & 7
    return (addr, 8 - k) if m.big_endian else (addr & ~7, k + 1)


def _span_dr(m: _M, addr: int) -> tuple[int, int]:
    k = addr & 7
    return (addr & ~7, k + 1) if m.big_endian else (addr, 8 - k)


def _e_ldl(m: _M, w: int) -> None:
    ea = S._addr_i(m, w)
    addr = m.data_address(ea, 1, store=False, span=_span_dl(m, ea))
    k, base, old = addr & 7, addr & ~7, m.read_reg(_rt(w))
    n = 8 - k if m.big_endian else k + 1
    v = m.load(addr if m.big_endian else base, n) << (8 * (8 - n))
    m.write_dreg(_rt(w), v | (old & ((1 << (8 * (8 - n))) - 1)))


def _e_ldr(m: _M, w: int) -> None:
    ea = S._addr_i(m, w)
    addr = m.data_address(ea, 1, store=False, span=_span_dr(m, ea))
    k, base, old = addr & 7, addr & ~7, m.read_reg(_rt(w))
    n = k + 1 if m.big_endian else 8 - k
    v = m.load(base if m.big_endian else addr, n)
    m.write_dreg(_rt(w), v | (old & (M64 ^ ((1 << (8 * n)) - 1))))


def _e_sdl(m: _M, w: int) -> None:
    ea = S._addr_i(m, w)
    addr = m.data_address(ea, 1, store=True, span=_span_dl(m, ea))
    k, base, r = addr & 7, addr & ~7, m.read_reg(_rt(w))
    n = 8 - k if m.big_endian else k + 1
    m.store(addr if m.big_endian else base, r >> (8 * (8 - n)), n, gather=False)


def _e_sdr(m: _M, w: int) -> None:
    ea = S._addr_i(m, w)
    addr = m.data_address(ea, 1, store=True, span=_span_dr(m, ea))
    k, base, r = addr & 7, addr & ~7, m.read_reg(_rt(w))
    if m.big_endian:                    # LS (k+1) bytes -> base..addr
        m.store(base, r, k + 1, gather=False)
    else:                               # LS (8-k) bytes -> addr..base+7
        m.store(addr, r, 8 - k, gather=False)


def _lld(m: _M, w: int, ea: int) -> None:
    """LLD: LL of a doubleword (the LLbit and the LL address as LL's)."""
    addr = m.data_address(ea, 8, store=False, atomic=True)
    m._linked = True
    try:
        m.write_dreg(_rt(w), m.load(addr, 8))
    finally:
        m._linked = False
    m.lr_addr, m.lr_valid = addr, True
    if m.system is not None:
        m.lr_pa = m._peek(addr)[0]


def _scd(m: _M, w: int, ea: int) -> None:
    """SCD: SC of a doubleword (rt ← 1 stored / 0 not)."""
    addr = m.data_address(ea, 8, store=True, atomic=True)
    ok = m.lr_valid and m.lr_addr == addr
    if ok:
        m.store(addr, m.read_reg(_rt(w)), 8, gather=False, atomic=True)
    m.write_dreg(_rt(w), int(ok))
    m.lr_valid = False


# -- Release 6 --------------------------------------------------------------------


def _e_dalign(m: _M, w: int) -> None:
    """DALIGN rd, rs, rt, bp: ``rt[63-8bp..0] || rs[63..64-8bp]``."""
    bp = (w >> 6) & 7
    a, b = m.read_reg(_rt(w)), m.read_reg(_rs(w))
    m.write_dreg(_rd(w), a if bp == 0 else (a << (8 * bp)) | (b >> (64 - 8 * bp)))


def _e_dbitswap(m: _M, w: int) -> None:
    v = m.read_reg(_rt(w))
    out = 0
    for i in range(8):
        out |= int(f"{(v >> (8 * i)) & 0xFF:08b}"[::-1], 2) << (8 * i)
    m.write_dreg(_rd(w), out)


def _e_dlsa(m: _M, w: int) -> None:
    m.write_dreg(_rd(w), (m.read_reg(_rs(w)) << (((w >> 6) & 3) + 1)) + m.read_reg(_rt(w)))


def _e_daui(m: _M, w: int) -> None:
    m.write_dreg(_rt(w), m.read_reg(_rs(w)) + (S._s16(w) << 16))


def _e_dahi(m: _M, w: int) -> None:
    m.write_dreg(_rs(w), m.read_reg(_rs(w)) + (S._s16(w) << 32))


def _e_dati(m: _M, w: int) -> None:
    m.write_dreg(_rs(w), m.read_reg(_rs(w)) + (S._s16(w) << 48))


def _e_ldpc(m: _M, w: int) -> None:
    """LDPC rs, offset: the doubleword at ``(PC & ~7) + sign_extend(offset
    << 3)``."""
    addr = ((m.pc & ~7) + (S._sx(w & 0x3FFFF, 18) << 3)) & M64
    m.write_dreg(_rs(w), m.load(m.data_address(addr, 8, store=False), 8, "pcload"))


def _e_lwupc(m: _M, w: int) -> None:
    """LWUPC rs, offset: the word at ``PC + sign_extend(offset << 2)``,
    zero-extended."""
    m.write_dreg(_rs(w), m.load(m.data_address(S._pc19(m, w), 4, store=False), 4, "pcload"))


# -- CP0 / CP1 moves ---------------------------------------------------------------


def _e_dmfc0(m: _M, w: int) -> None:
    """DMFC0: a 64-bit register whole; a 32-bit one sign-extended (as
    QEMU: the manual does not say, sim.DEVIATIONS).  EntryLo with 64-bit
    addresses: RI / XI (kept at 31 / 30, MIPS32's place) at 63 / 62."""
    reg, sel = S._cp0_cell(m, w, "DMFC0")
    m.write_dreg(_rt(w), m.cp0_dread(reg, sel))


def _e_dmtc0(m: _M, w: int) -> None:
    """DMTC0; EntryLo with 64-bit addresses: RI / XI from bits 63 / 62
    (Vol. III 9.6), the PFN from 29:6 (PFNX and the bits above PABITS:
    0)."""
    reg, sel = S._cp0_cell(m, w, "DMTC0")
    m.cp0_dwrite(reg, sel, m.read_reg(_rt(w)))


def _e_dmfc1(m: _M, w: int) -> None:
    m.write_dreg(_rt(w), S.fpr_get(m, S._fs(w), "L"))


def _e_dmtc1(m: _M, w: int) -> None:
    S.fpr_set(m, S._fs(w), "L", m.read_reg(_rt(w)))


def _none(fn):
    """The doubleword executors return nothing (no branch target)."""
    def ex(m: _M, w: int) -> int | None:
        fn(m, w)
        return None
    ex.__name__ = fn.__name__ if hasattr(fn, "__name__") else "ex"
    return ex


EXECUTORS: dict[str, S.Executor] = {name: _none(fn) for name, fn in {
    "DADD": _e_dadd, "DADDU": _e_daddu, "DSUB": _e_dsub, "DSUBU": _e_dsubu,
    "DADDI": _e_daddi, "DADDIU": _e_daddiu,
    "DCLO": _e_dclo, "DCLZ": _e_dclz, "DCLO@R6": _e_dclo, "DCLZ@R6": _e_dclz,
    "DEXT": _dext(""), "DEXTM": _dext("M"), "DEXTU": _dext("U"),
    "DINS": _dins(""), "DINSM": _dins("M"), "DINSU": _dins("U"),
    "DSBH": _e_dsbh, "DSHD": _e_dshd,
    "DROTR": _e_drotr, "DROTR32": _e_drotr32, "DROTRV": _e_drotrv,
    "DSLL": _shift("l"), "DSLL32": _shift("l", 32), "DSLLV": _shift("l", variable=True),
    "DSRL": _shift("r"), "DSRL32": _shift("r", 32), "DSRLV": _shift("r", variable=True),
    "DSRA": _shift("a"), "DSRA32": _shift("a", 32), "DSRAV": _shift("a", variable=True),
    "DMULT": _e_dmult, "DMULTU": _e_dmultu, "DDIV": _e_ddiv, "DDIVU": _e_ddivu,
    **_r6_dmuldiv(),
    "LD": _e_ld, "LWU": _e_lwu, "SD": _e_sd,
    "LDL": _e_ldl, "LDR": _e_ldr, "SDL": _e_sdl, "SDR": _e_sdr,
    "LLD": lambda m, w: _lld(m, w, S._addr_i(m, w)),
    "SCD": lambda m, w: _scd(m, w, S._addr_i(m, w)),
    "LLD@R6": lambda m, w: _lld(m, w, S._addr_e(m, w)),
    "SCD@R6": lambda m, w: _scd(m, w, S._addr_e(m, w)),
    "DALIGN": _e_dalign, "DBITSWAP": _e_dbitswap, "DLSA": _e_dlsa,
    "DAUI": _e_daui, "DAHI": _e_dahi, "DATI": _e_dati,
    "LDPC": _e_ldpc, "LWUPC": _e_lwupc,
    "DMFC0": _e_dmfc0, "DMTC0": _e_dmtc0, "DMFC1": _e_dmfc1, "DMTC1": _e_dmtc1,
}.items()}
