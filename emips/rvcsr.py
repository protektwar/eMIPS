"""The RISC-V hart's CSRs (PLAN M25 phases 1c / 2a; The RISC-V Instruction
Set Manual, Volume II: Privileged Architecture 20250508 — "priv." — and
Volume I's Zicsr / Zicntr / Zihpm / F chapters).

:class:`CSRFile` holds the CSRs of one :class:`emips.simrv.RVMachine`:
:data:`CSRS` lists every one eMIPS implements with the condition under
which it exists (the hart's modes — M, M + U, M + S + U —, its XLEN and
extensions), how it reads and how it is written (WARL / WLRL legalised
here).  :meth:`CSRFile.check` applies priv. 2.1: a CSR that does not
exist, needs more privilege (number bits 9:8), is read-only (bits 11:10 =
11) and written, a counter its counter-enable bits hide (3.1.11 /
4.1.5), satp / SFENCE.VMA with mstatus.TVM, or fflags / frm / fcsr with
mstatus.FS = Off is an illegal instruction.

Where the specification leaves a choice, eMIPS takes Spike's (the
reference simulator, ``riscv/csr_init.cc`` / ``csrs.cc``): MPP / SPP
legalise an unsupported mode to the least privileged one; mtvec / stvec
drop MODE bit 1 (2 / 3 are reserved); mcounteren / scounteren /
mcountinhibit have one bit per implemented counter; mideleg delegates
SSI / STI / SEI, medeleg every exception S-mode can take (not the M-mode
ECALL; instruction-address-misaligned only without C); mip's SSIP / STIP
/ SEIP are writable with S-mode, SEIP read = the written bit OR the
external line; misa is read-only; mvendorid / marchid / mimpid read 0;
mhpmcounter3-31 / mhpmevent3-31 (and with Zihpm hpmcounter3-31) read 0.
The trigger registers reach :mod:`emips.rvdebug`; dcsr / dpc / dscratch0-1
exist only in Debug Mode (Debug spec 4.9).

**The hypervisor extension** (LF7, priv. ch. 21): the h* CSRs, the VS
CSRs and mtval2 / mtinst exist with ``h``.  With V = 1 the supervisor
CSRs sstatus, sie, stvec, sscratch, sepc, scause, stval, sip and satp are
their VS twins (:data:`VS_ALIAS`; :meth:`CSRFile.check` judges the
access by the number the instruction names).  :meth:`CSRFile.check`
raises the virtual-instruction exception (22) where 21.6.1 says: a CSR
of supervisor or hypervisor level from VS / VU-mode, a counter hcounteren
(or scounteren in VU) hides while mcounteren allows it, satp with
hstatus.VTVM; FS / VS = Off in either status register is always an
illegal instruction.  Spike's choices again where the manual leaves
them: mideleg's VS bits and SGEI read 1, hie.SGEIE writable, GEILEN = 0
(hgeip / hgeie / VGEIN read 0 until LF9's IMSIC), VMIDLEN = VMIDMAX,
hgatp's MODE kept on an unsupported value (its other fields written),
vsatp likewise (WARL) — but a write of satp with V = 1 and an
unsupported MODE does nothing; hedeleg's writable bits are Table 39's
"writable" ones plus bit 0 (Spike: also with C) and 19 with Zicntr
(medeleg's too).
Choices: ``sim.DEVIATIONS["RISC-V"]``, ``["RISC-V hypervisor"]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from .rvdebug import DCSR_DEBUGVER, MPTE, MTE, TINFO
from .rvhpm import inhibit_bit as _inh

if TYPE_CHECKING:
    from .simrv import RVMachine

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF

# mstatus (priv. 3.1.6)
SIE, MIE, SPIE, UBE, MPIE, SPP = 1, 3, 5, 6, 7, 8
VS_SHIFT, MPP_SHIFT, FS_SHIFT, XS_SHIFT = 9, 11, 13, 15
MPRV, SUM, MXR, TVM, TW, TSR = 17, 18, 19, 20, 21, 22
UXL_SHIFT, SXL_SHIFT = 32, 34
#: mip / mie bits (priv. 3.1.9)
SSI, MSI, STI, MTI, SEI, MEI = 1, 3, 5, 7, 9, 11
#: the interrupt priority (3.1.9: MEI, MSI, MTI, SEI, SSI, STI)
PRIORITY = (MEI, MSI, MTI, SEI, SSI, STI)
S_INTERRUPTS = (1 << SSI) | (1 << STI) | (1 << SEI)
M_INTERRUPTS = (1 << MSI) | (1 << MTI) | (1 << MEI)
# the hypervisor extension (priv. ch. 21)
#: mstatus MPV / GVA (21.4.1; RV32: mstatush bits 7 / 6)
MPV, GVA = 39, 38
#: the VS-level and guest external interrupts (21.2.3, Table 43)
VSSI, VSTI, VSEI, SGEI = 2, 6, 10, 12
VS_INTERRUPTS = (1 << VSSI) | (1 << VSTI) | (1 << VSEI)
HS_INTERRUPTS = VS_INTERRUPTS | (1 << SGEI)
#: the priority with H (21.2.3: ... SEI, SSI, STI, SGEI, VSEI, VSSI, VSTI)
PRIORITY_H = PRIORITY + (SGEI, VSEI, VSSI, VSTI)
#: hstatus (21.2.1)
H_GVA, H_SPV, H_SPVP, H_HU, H_VTVM, H_VTW, H_VTSR = 6, 7, 8, 9, 20, 21, 22
HSTATUS_MASK = sum(1 << b for b in (H_GVA, H_SPV, H_SPVP, H_HU, H_VTVM, H_VTW, H_VTSR))
#: hstatus.VGEIN (21.2.1; with Ssaia: the guest interrupt file, AIA 3.7)
VGEIN_SHIFT, VGEIN_MASK = 12, 0x3F << 12
#: the supervisor CSRs whose VS twins V = 1 accesses instead (21.2)
VS_ALIAS = {0x100: 0x200, 0x104: 0x204, 0x105: 0x205, 0x140: 0x240, 0x141: 0x241,
            0x142: 0x242, 0x143: 0x243, 0x144: 0x244, 0x180: 0x280,
            0x14D: 0x24D, 0x15D: 0x25D,                 # (Sstc 19.2.1: vstimecmp)
            0x150: 0x250, **{0x150 + k: 0x250 + k for k in (1, 2, 3, 5, 6, 7)},   # (5.4)
            0x15C: 0x25C, 0xDB0: 0xEB0, 0x114: 0x214, 0x154: 0x254}    # (AIA 2.3)
#: the virtual-instruction exception (21.6.1)
VIRTUAL_INSTRUCTION = 22
# LF8: the privileged additions
#: menvcfg / henvcfg / senvcfg (3.1.18, 21.2.5, 10.1.4; unpriv. 19.5)
FIOM, CBIE, CBCFE, CBZE = 1, 3 << 4, 1 << 6, 1 << 7
CDE, ADUE, PBMTE, STCE = 1 << 60, 1 << 61, 1 << 62, 1 << 63
#: the stateen bits (4.2): C, FCSR, JVT; P1P13, CONTEXT, CSRIND, ENVCFG, SE0
SE_C, SE_FCSR, SE_JVT = 1, 2, 4
SE_P1P13, SE_CONTEXT, SE_CSRIND, SE_ENVCFG, SE0 = 1 << 56, 1 << 57, 1 << 60, 1 << 62, 1 << 63
#: the AIA's state-enable bits (AIA 2.5): 58 the IMSIC's state, 59 the rest
SE_IMSIC, SE_AIA = 1 << 58, 1 << 59
#: mnstatus (8.3): NMIE, MNPV, MNPP 12:11
NMIE, MNPV, MNPP_SHIFT = 1 << 3, 1 << 7, 11
#: mseccfg (6.2; 31.5.3): MML, MMWP, RLB, USEED, SSEED
MML, MMWP, RLB, USEED, SSEED = 1, 2, 4, 1 << 8, 1 << 9
#: the local counter-overflow interrupt (Sscofpmf, 20.1)
LCOFI = 13
# LF10: control-flow integrity (priv. ch. 22, 3.1.6.8, 3.1.18, 3.1.19; unpriv. ch. 33)
#: mstatus / sstatus / vsstatus SPELP, mstatus MPELP (RV32: mstatush bit 9)
SPELP, MPELP = 23, 41
#: xenvcfg LPE / SSE, mseccfg MLPE, mnstatus MNPELP, dcsr pelp
LPE, SSE, MLPE, MNPELP, DCSR_PELP = 1 << 2, 1 << 3, 1 << 10, 1 << 9, 1 << 18
#: the software-check exception and its xtval codes (3.1.16; 22.1, 22.2)
SOFTWARE_CHECK, LANDING_PAD_FAULT, SHADOW_STACK_FAULT = 18, 2, 3
#: Smcdeleg's selects (9.1, Table 20): siselect 0x40 + counter
DELEG_FIRST, DELEG_LAST = 0x40, 0x5F


@dataclass(frozen=True)
class CSR:
    """One CSR: ``exists(c)`` on this hart, ``read(c)``, ``write(c, v)``
    (None: read-only)."""
    num: int
    name: str
    exists: Callable[["CSRFile"], bool]
    read: Callable[["CSRFile"], int]
    write: Callable[["CSRFile", int], None] | None = None


class CSRFile:
    """The CSRs of one hart (see the module docstring)."""

    def __init__(self, m: "RVMachine") -> None:
        self.m = m
        #: the hart's privilege modes (3 M, 1 S, 0 U)
        self.modes = frozenset(m.modes)
        self.has_s = 1 in self.modes
        self.has_u = 0 in self.modes
        #: the least-privileged mode (an xRET's xPP afterwards, MPP's
        #: legalisation)
        self.lowest_priv = min(self.modes)
        self.zihpm = "zihpm" in m.extensions
        # mstatus: reset MPP = the lowest mode (Spike), UXL / SXL = XLEN
        self.mstatus = self.lowest_priv << MPP_SHIFT
        if m.xlen == 64:
            if self.has_u:
                self.mstatus |= 2 << UXL_SHIFT
            if self.has_s:
                self.mstatus |= 2 << SXL_SHIFT
        self.mtvec = self.stvec = 0
        self.mepc = self.sepc = 0
        self.mcause = self.scause = 0
        self.mtval = self.stval = 0
        self.mscratch = self.sscratch = 0
        #: Zcmt's table jump base (unpriv. 28.14.3; mode 0 only)
        self.jvt = 0
        #: mseccfg (priv. 3.1.17): Zkr's SSEED (9) / USEED (8)
        self.mseccfg = 0
        #: Zkr's entropy source: the polls so far (unpriv. 31.5)
        self.seed_polls = 0
        self.mie = 0
        #: mip's software-writable bits (SSIP, STIP, SEIP's own bit)
        self.mip_sw = 0
        self.medeleg = self.mideleg = 0
        self.mcounteren = self.scounteren = 0
        self.mcountinhibit = 0
        self.menvcfg = self.senvcfg = 0
        self.satp = 0
        #: mcycle / minstret (64-bit on every XLEN, priv. 3.1.10)
        self.mcycle = 0
        self.minstret = 0
        #: Sdext (Debug spec 4.9): dcsr (debugver 4, prv 3 at reset), dpc,
        #: dscratch0 / 1 — Debug Mode only
        self.dcsr = DCSR_DEBUGVER | 3
        self.dpc = 0
        self.dscratch0 = self.dscratch1 = 0
        #: the interrupt priority, the S-level interrupts (sie / sip /
        #: mideleg), the mip bits the platform's lines may reach, the
        #: software-writable mip bits beyond SSIP / STIP / SEIP
        self.priority = PRIORITY
        self.s_interrupts = S_INTERRUPTS
        self.line_mask = M_INTERRUPTS | (1 << SEI) if self.has_s else M_INTERRUPTS
        self.mip_extra = 0
        #: the hypervisor extension (21.2): hstatus (VSXL = HSXLEN, read-
        #: only), the delegations, hvip's VSEIP / VSTIP (its VSSIP is mip's,
        #: in mip_sw), the guest's counters / time / trap values / hgatp,
        #: the VS CSRs (vsstatus.UXL = VSXLEN, read-only), mtval2 / mtinst
        self.has_h = "h" in m.extensions and self.has_s
        wide = (2 << 32) if m.xlen == 64 else 0
        self.hstatus = wide
        self.hedeleg = self.hideleg = 0
        self.hvip = 0
        self.hcounteren = 0
        self.htimedelta = 0
        self.htval = self.htinst = 0
        self.hgatp = 0
        self.henvcfg = 0
        self.vsstatus = wide
        self.vstvec = self.vsscratch = self.vsepc = self.vscause = self.vstval = 0
        self.vsatp = 0
        self.mtval2 = self.mtinst = 0
        if self.has_h:                          # (21.4.2: VS interrupts always delegated)
            self.mideleg = HS_INTERRUPTS
            self.priority = PRIORITY_H
        # LF8: the privileged additions (see the module docstring)
        e = m.extensions
        generic = m.p8700 is None
        self.sstc = "sstc" in e and self.has_s
        self.sscofpmf = "sscofpmf" in e and generic
        self.smcntrpmf = "smcntrpmf" in e
        self.stateen = "smstateen" in e
        self.smrnmi = "smrnmi" in e
        self.smepmp = "smepmp" in e
        #: Smaia / Ssaia (AIA ch. 2; Smaia brings Ssaia with S-mode and
        #: Smcsrind / Sscsrind with them: Spike's isa_parser)
        self.smaia = "smaia" in e and generic
        self.ssaia = ("ssaia" in e or self.smaia) and self.has_s and generic
        self.csrind = "smcsrind" in e or self.smaia
        self.sscsrind = ("sscsrind" in e or self.ssaia) and self.has_s
        self.cdeleg = "smcdeleg" in e and self.has_s
        #: Zicfilp / Zicfiss (LF10): the expected-landing-pad state (22.1.1:
        #: NO_LP_EXPECTED at reset) and the shadow stack pointer (33.2.2)
        self.zicfilp = "zicfilp" in e and generic
        self.zicfiss = "zicfiss" in e and generic
        self.elp = 0
        self.ssp = 0
        #: hgeie (21.2.4: GEILEN bits from bit 1 — the IMSIC's guest files)
        self.hgeie = 0
        #: the generic hart's counters (Zihpm: emips.rvhpm; the P8700: its own)
        self.hpm = None
        if self.zihpm and generic:
            from .rvhpm import HPM
            self.hpm = HPM(m)
        #: mstateen0-3 / hstateen0-3 / sstateen0-3 (4.1)
        self.mstateen = [0] * 4
        self.hstateen = [0] * 4
        self.sstateen = [0] * 4
        #: Sstc (19): stimecmp / vstimecmp (64 bits; Spike resets them to 0)
        self.stimecmp = self.vstimecmp = 0
        #: Smrnmi (8.3): mncause = an interrupt, no RNMI causes (Spike's
        #: value until an RNMI writes its own); NMIE = 0 at reset
        self.mnscratch = self.mnepc = 0
        self.mncause = 1 << (m.xlen - 1)
        self.mnstatus = 0
        #: Smcntrpmf (7.2.1): mcyclecfg / minstretcfg
        self.mcyclecfg = self.minstretcfg = 0
        #: Smcsrind / Sscsrind (5.2-5.4)
        self.miselect = self.siselect = self.vsiselect = 0
        if self.sscofpmf:                       # (20.1: LCOFI = mip / mie bit 13, after STI
            lcofi = 1 << LCOFI                  # — 3.1.9 —, after VSTI with H as Spike)
            self.mip_extra = lcofi
            self.s_interrupts |= lcofi if self.has_s else 0
            self.priority = self.priority + (LCOFI,)
        self.aia = None
        if self.smaia or self.ssaia:
            from .rvaia import AIA
            self.aia = AIA(self, self.smaia)
        self.table = {c.num: c for c in CSRS if c.exists(self)}
        if m.p8700 is not None:                 # (the P8700's own: emips.p8700)
            from . import p8700 as P
            self.table.update({c.num: c for c in P.CSRS if c.exists(self)})
            self.priority = P.PRIORITY
            lcofi = 1 << P.LCOFI
            self.s_interrupts |= lcofi if self.has_s else 0
            self.line_mask |= P.LINE_BITS
            self.mip_extra = lcofi

    # -- mstatus helpers -------------------------------------------------------------

    def status_bit(self, bit: int) -> int:
        return (self.mstatus >> bit) & 1

    def status_field(self, shift: int, mask: int) -> int:
        return (self.mstatus >> shift) & mask

    def legal_mode(self, mode: int) -> int:
        """A privilege mode an xPP field may hold (Spike's legalisation:
        an unsupported mode becomes the least privileged one)."""
        return mode if mode in self.modes else self.lowest_priv

    def set_status(self, **fields: int) -> None:
        """Set mstatus bits by name: mie mpie sie spie mprv (one bit), mpp
        (2 bits, legalised), spp (one bit)."""
        s = self.mstatus
        bits = {"mie": MIE, "mpie": MPIE, "sie": SIE, "spie": SPIE, "mprv": MPRV, "spp": SPP}
        for name, v in fields.items():
            if name == "mpp":
                s = (s & ~(3 << MPP_SHIFT)) | (self.legal_mode(v) << MPP_SHIFT)
            else:
                s = (s & ~(1 << bits[name])) | ((v & 1) << bits[name])
        self.mstatus = s

    def mstatus_value(self) -> int:
        """mstatus as read (SD = FS or XS Dirty)."""
        s = self.mstatus
        if (s >> FS_SHIFT) & 3 == 3 or (s >> VS_SHIFT) & 3 == 3:     # (30.3.2: VS too)
            s |= 1 << (self.m.xlen - 1) if self.m.xlen == 32 else 1 << 63
        return s

    def mstatus_mask(self) -> int:
        """mstatus's writable bits on this hart."""
        mask = (1 << MIE) | (1 << MPIE) | (3 << MPP_SHIFT)
        if self.has_s:
            mask |= (1 << SIE) | (1 << SPIE) | (1 << SPP) | (1 << TSR)
        if self.has_u:
            mask |= (1 << MPRV) | (1 << TW)
        if self.has_s and self.m.paging:
            mask |= (1 << SUM) | (1 << MXR) | (1 << TVM)
        if self.m.flen:
            mask |= 3 << FS_SHIFT
        if self.m.vector is not None:
            mask |= 3 << VS_SHIFT
        if self.has_h:
            mask |= (1 << MPV) | (1 << GVA)
        if self.zicfilp:                        # (3.1.6.8: the previous ELP)
            mask |= (1 << MPELP) | ((1 << SPELP) if self.has_s else 0)
        return mask

    def write_mstatus(self, v: int, mask: int | None = None) -> None:
        mask = self.mstatus_mask() if mask is None else mask
        mpp = self.legal_mode((v >> MPP_SHIFT) & 3)
        v = (v & ~(3 << MPP_SHIFT)) | (mpp << MPP_SHIFT)
        self.mstatus = (self.mstatus & ~mask) | (v & mask)

    def sstatus_mask(self) -> int:
        """sstatus's writable bits (priv. 10.1.1)."""
        mask = (1 << SIE) | (1 << SPIE) | (1 << SPP)
        if self.m.paging:
            mask |= (1 << SUM) | (1 << MXR)
        if self.m.flen:
            mask |= 3 << FS_SHIFT
        if self.m.vector is not None:
            mask |= 3 << VS_SHIFT
        if self.zicfilp:                        # (10.1.1.4: SPELP)
            mask |= 1 << SPELP
        return mask

    def sstatus_value(self) -> int:
        read = self.sstatus_mask() | (3 << XS_SHIFT) | (1 << UBE)
        if self.m.xlen == 64:
            read |= 3 << UXL_SHIFT | 1 << 63
        else:
            read |= 1 << 31
        return self.mstatus_value() & read

    def vsstatus_value(self) -> int:
        """vsstatus as read (21.2.11): sstatus's fields, SD from its own FS /
        VS, UXL = VSXLEN."""
        s = self.vsstatus
        if (s >> FS_SHIFT) & 3 == 3 or (s >> VS_SHIFT) & 3 == 3:
            s |= 1 << (self.m.xlen - 1)
        return s

    # -- the extension context status (FS / VS: 3.1.6.6, 21.2.11) ----------------------

    def fs_on(self) -> bool:
        """F's state usable: mstatus.FS (and with V = 1 vsstatus.FS) not Off."""
        if not (self.mstatus >> FS_SHIFT) & 3:
            return False
        return not self.m.v or bool((self.vsstatus >> FS_SHIFT) & 3)

    def vs_on(self) -> bool:
        """The vector state usable: mstatus.VS (and with V = 1 vsstatus.VS)."""
        if not (self.mstatus >> VS_SHIFT) & 3:
            return False
        return not self.m.v or bool((self.vsstatus >> VS_SHIFT) & 3)

    def fs_dirty(self) -> None:
        """F's state written: FS = Dirty (with V = 1 in both registers)."""
        self.mstatus |= 3 << FS_SHIFT
        if self.m.v:
            self.vsstatus |= 3 << FS_SHIFT

    def vs_dirty(self) -> None:
        self.mstatus |= 3 << VS_SHIFT
        if self.m.v:
            self.vsstatus |= 3 << VS_SHIFT

    # -- interrupts ------------------------------------------------------------------

    def mip(self) -> int:
        """mip as read: the platform's lines (MSIP, MTIP, MEIP, the
        external SEIP), the software-writable bits and hvip's VSEIP /
        VSTIP (21.2.3: hip.VSEIP / VSTIP = hvip's, GEILEN = 0); with Sstc
        and menvcfg.STCE, STIP = time >= stimecmp, with henvcfg.STCE too
        VSTIP |= time + htimedelta >= vstimecmp (19.1.2 / 19.2.2)."""
        v = self.m.irq_lines & self.line_mask | self.mip_sw | self.hvip
        a = self.aia
        if a is not None:
            if a.mvien >> SEI & 1:              # (AIA 5.3: SEIP = the controller's only)
                v = (v & ~(1 << SEI)) | (self.m.irq_lines & (1 << SEI))
            im = self.imsic()
            if im is not None and self.has_h:   # (AIA 3.10 / 21.2.3: the VGEIN file's
                f = im.guests.get(self.vgein())  # signal into VSEIP, SGEIP = hgeip & hgeie)
                if f is not None and f.signal():
                    v |= 1 << VSEI
                if im.hgeip & self.hgeie:
                    v |= 1 << SGEI
        if self.sstc and self.menvcfg & STCE:
            t = self.m.mtime
            v = (v & ~(1 << STI)) | int(t >= self.stimecmp) << STI
            if self.has_h and self.henv() & STCE:
                v = (v & ~(1 << VSTI)) | self.hvip & (1 << VSTI) \
                    | int((t + self.htimedelta) & M64 >= self.vstimecmp) << VSTI
        return v

    def henv(self) -> int:
        """henvcfg as read: PBMTE / STCE / ADUE / SSE read 0 while
        menvcfg's are 0 (21.2.5; Spike's henvcfg_csr_t)."""
        return self.henvcfg & (self.menvcfg | ~(PBMTE | STCE | ADUE | SSE))

    def timer_signals(self) -> int:
        """Sstc's STIP / VSTIP as the comparators give them now."""
        t = self.m.mtime
        return int(t >= self.stimecmp) << STI \
            | int((t + self.htimedelta) & M64 >= self.vstimecmp) << VSTI

    def mip_mask(self) -> int:
        mask = (S_INTERRUPTS if self.has_s else 0) | self.mip_extra \
            | ((1 << VSSI) if self.has_h else 0)
        if self.sstc and self.menvcfg & STCE:
            mask &= ~(1 << STI)                 # (19.1.2: read-only, the comparator's)
        return mask

    def mie_mask(self) -> int:
        return M_INTERRUPTS | (S_INTERRUPTS if self.has_s else 0) | self.line_mask \
            | self.mip_extra | (HS_INTERRUPTS if self.has_h else 0)

    def medeleg_mask(self) -> int:
        mask = sum(1 << c for c in (1, 2, 3, 4, 5, 6, 7, 8, 9))
        if self.m.ialign == 4:
            mask |= 1                           # (with C it cannot happen: Spike)
        if self.m.paging:
            mask |= (1 << 12) | (1 << 13) | (1 << 15)
        if "zicntr" in self.m.extensions:
            mask |= 1 << 19                     # (hardware error: Spike, with Zicntr)
        if self.has_h:                          # (21.6.1: VS ecall, the guest-page
            mask |= sum(1 << c for c in (10, 20, 21, 22, 23))    # faults, virtual insn)
        if self.zicfilp or self.zicfiss:        # (the software check: Spike)
            mask |= 1 << SOFTWARE_CHECK
        return mask

    def hedeleg_mask(self) -> int:
        """Table 39's writable bits (bit 0 also with C, 19 with Zicntr: Spike)."""
        mask = sum(1 << c for c in (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 15))
        if "zicntr" in self.m.extensions:
            mask |= 1 << 19
        if self.zicfilp or self.zicfiss:        # (Spike's hedeleg mask)
            mask |= 1 << SOFTWARE_CHECK
        return mask

    # -- control-flow integrity (LF10) ---------------------------------------------

    def senv(self) -> int:
        """senvcfg as read: SSE reads 0 while menvcfg.SSE is 0, and with V = 1
        while henvcfg.SSE is 0 (22.2.2, 21.2.5; Spike's senvcfg_csr_t)."""
        gate = self.henv() if self.m.v else self.menvcfg
        return self.senvcfg & ~(0 if gate & SSE else SSE)

    def xlpe(self, priv: int, v: int) -> bool:
        """Zicfilp enabled in a mode (Tables 51 / 52)."""
        if not self.zicfilp:
            return False
        if priv == 3:
            return bool(self.mseccfg & MLPE)
        if priv == 1:
            return bool((self.henv() if v else self.menvcfg) & LPE)
        return bool((self.senvcfg if self.has_s else self.menvcfg) & LPE)

    def xsse(self) -> bool:
        """Zicfiss active in the current mode (Table 53: never in M, never
        without S-mode; S / HS menvcfg.SSE, VS henvcfg.SSE, U / VU
        senvcfg.SSE as read — Spike's xSSE)."""
        m = self.m
        if not self.zicfiss or m.priv == 3 or not self.has_s or not self.menvcfg & SSE:
            return False
        if m.v and not self.henv() & SSE:
            return False
        return m.priv != 0 or bool(self.senv() & SSE)

    def sse_check(self, insn: int) -> None:
        """ssp's access rule (22.2.1) and SSAMOSWAP's (33.2.7, Listing 9):
        Spike's ``require_envcfg(SSE)``."""
        from .simrv import ILLEGAL, RVTrap
        m = self.m
        if m.priv != 3 and not self.menvcfg & SSE:
            raise RVTrap(ILLEGAL, insn)
        if self.has_s and m.priv == 0 and not m.v and not self.senv() & SSE:
            raise RVTrap(ILLEGAL, insn)
        if m.v and (not self.henv() & SSE or (m.priv == 0 and not self.senv() & SSE)):
            raise RVTrap(VIRTUAL_INSTRUCTION, insn)

    def counter_mask(self) -> int:
        if self.m.p8700 is not None:            # (four counters: MD01502 11.1)
            return 0x7F
        return 0x7 | (0xFFFFFFF8 if self.zihpm else 0)

    # -- access --------------------------------------------------------------------

    def exists(self, num: int) -> bool:
        return num in self.table

    def rebuild(self) -> None:
        """The table again: a platform gave the hart an IMSIC (the CSRs
        that exist only with one: mtopei, stopei, vstopei)."""
        keep = {n: c for n, c in self.table.items() if c not in CSRS}
        self.table = {c.num: c for c in CSRS if c.exists(self)}
        self.table.update(keep)

    def vgein(self) -> int:
        """hstatus.VGEIN (0 without Ssaia)."""
        return (self.hstatus & VGEIN_MASK) >> VGEIN_SHIFT

    def imsic(self):
        return getattr(self.m, "imsic", None)

    def check(self, num: int, *, write: bool, insn: int) -> None:
        """priv. 2.1 and the per-CSR rules (see the module docstring); with
        H, 21.6.1's virtual-instruction exception (Spike's order: a write
        of a read-only CSR, then the privilege — S- and H-level CSRs from
        VS / VU-mode — then each CSR's own rule; the state-enable rules of
        the CSRs Spike checks before the privilege come first)."""
        from .simrv import ILLEGAL, RVTrap
        m = self.m
        if num not in self.table or (write and num >> 10 == 3):
            raise RVTrap(ILLEGAL, insn)
        pre = _PRE_CHECKS.get(num)
        if pre is not None:                     # (Spike: before the privilege)
            pre(self, num, write, insn)
        need, v = (num >> 8) & 3, m.v
        prv = 2 if m.priv == 1 and not v and self.has_h else m.priv     # (HS = 2)
        if need > prv:
            raise RVTrap(VIRTUAL_INSTRUCTION if v and need <= 2 else ILLEGAL, insn)
        if num == 0x015:                        # seed (unpriv. 31.5.1 / 31.5.3)
            if not write:                       # (a read-only access: illegal)
                raise RVTrap(ILLEGAL, insn)
            if v:                               # (31.5.3: VS / VU, as Spike)
                raise RVTrap(VIRTUAL_INSTRUCTION if self.mseccfg >> 9 & 1 else ILLEGAL, insn)
            if (m.priv == 1 and not self.mseccfg >> 9 & 1) or \
                    (m.priv == 0 and not self.mseccfg >> 8 & 1):
                raise RVTrap(ILLEGAL, insn)
        if 0x7B0 <= num <= 0x7BF and not m.debug_mode:
            raise RVTrap(ILLEGAL, insn)         # Debug spec 4.9: Debug Mode only
        if num in (0x001, 0x002, 0x003):
            self.fp_check(insn)
        if num in VECTOR_CSRS and not self.vs_on():
            raise RVTrap(ILLEGAL, insn)         # 30.3.2: VS = Off
        if 0xC00 <= num <= 0xC1F or 0xC80 <= num <= 0xC9F:
            self.counter_check(num & 31, insn)
        if num == 0x180:
            if v:
                if self.hstatus >> H_VTVM & 1:  # 21.2.1: VTVM
                    raise RVTrap(VIRTUAL_INSTRUCTION, insn)
            elif m.priv == 1 and self.status_bit(TVM):
                raise RVTrap(ILLEGAL, insn)     # 3.1.6.6: TVM
        if num == 0x680 and m.priv == 1 and self.status_bit(TVM):
            raise RVTrap(ILLEGAL, insn)         # 21.2.10: hgatp with TVM in HS-mode
        if num == 0x017:                        # jvt (4.2: JVT, a user-level bit)
            self.stateen_check(SE_JVT, insn, user=True)
        if num == 0x120:                        # scountinhibit (9.2)
            if not self.menvcfg & CDE:
                raise RVTrap(ILLEGAL, insn)
            if v:                               # (9.2: VS / VU — Spike lets VS through)
                raise RVTrap(VIRTUAL_INSTRUCTION, insn)
        if num == 0xDA0 and v and self.cdeleg and self.menvcfg & CDE:
            raise RVTrap(VIRTUAL_INSTRUCTION, insn)     # 9.3: scountovf
        if num in _INDIRECT:                    # (Spike: after the privilege)
            self.indirect_check(num, write, insn)
        if self.aia is not None:
            self.aia_check(num, write, insn)
        if num == 0xC01 and m.p8700 is not None:
            m.p8700.read_time(insn)             # (the P8700 has no time: MD01502 7.2)

    def aia_check(self, num: int, write: bool, insn: int) -> None:
        """The AIA's rules after the privilege (AIA 2.3 / 6.3.2): with V = 1
        hvictl.VTI makes sip / sie (siph / sieh) and a stimecmp write a
        virtual instruction; *topei of a guest file needs a valid VGEIN."""
        from .simrv import ILLEGAL, RVTrap
        from .rvaia import VTI
        m = self.m
        if m.v and self.aia.hvictl & VTI and (
                num in (0x144, 0x104, 0x154, 0x114) or (write and num in (0x14D, 0x15D))):
            raise RVTrap(VIRTUAL_INSTRUCTION, insn)
        if num == 0x15C and m.v or num == 0x25C:
            im = self.imsic()
            if im is None or self.vgein() not in im.guests:
                raise RVTrap(VIRTUAL_INSTRUCTION if m.v else ILLEGAL, insn)

    def counter_check(self, n: int, insn: int) -> None:
        """3.1.11 / 10.1.5 / 21.2.6: counter ``n``'s enables (Spike's
        counter_proxy_csr_t)."""
        from .simrv import ILLEGAL, RVTrap
        m, bit = self.m, 1 << n
        if m.priv < 3 and not self.mcounteren & bit:
            raise RVTrap(ILLEGAL, insn)
        if m.v and not self.hcounteren & bit:
            raise RVTrap(VIRTUAL_INSTRUCTION, insn)
        if m.priv == 0 and self.has_s and not self.scounteren & bit:
            raise RVTrap(VIRTUAL_INSTRUCTION if m.v else ILLEGAL, insn)

    def fp_check(self, insn: int) -> None:
        """fflags / frm / fcsr — and with Zfinx every FP instruction (Spike's
        require_fp): FS = Off (F 20.2 / priv. 3.1.6.6; with V = 1 either FS:
        always illegal, 21.6.1; Zfinx has no FS: unpriv. 26.6), then with
        Smstateen and Zfinx the FCSR bit (4.2)."""
        from .simrv import ILLEGAL, RVTrap
        if self.m.flen and not self.fs_on():
            raise RVTrap(ILLEGAL, insn)
        if getattr(self.m, "inx", False):
            self.stateen_check(SE_FCSR, insn, user=True)

    def stateen_check(self, bit: int, insn: int, *, user: bool = False,
                      index: int = 0) -> None:
        """The state-enable rule (4.1): below M, mstateen[index]'s bit; with
        V = 1, hstateen's (a virtual-instruction exception); for user-level
        state (``user``) in U / VU, sstateen's too."""
        if not self.stateen:
            return
        from .simrv import ILLEGAL, RVTrap
        m = self.m
        if m.priv < 3 and not self.mstateen[index] & bit:
            raise RVTrap(ILLEGAL, insn)
        if m.v and not self.hstateen[index] & bit:
            raise RVTrap(VIRTUAL_INSTRUCTION, insn)
        if user and m.priv == 0 and self.has_s and not self.sstateen[index] & bit:
            raise RVTrap(VIRTUAL_INSTRUCTION if m.v else ILLEGAL, insn)

    def indirect_check(self, num: int, write: bool, insn: int) -> None:
        """mireg* / sireg* / vsireg* (5.2-5.5, 9.1; AIA 2.1-2.5): what the
        select reaches — Smcdeleg's counters 0x40-0x5F, the AIA's iprio
        arrays 0x30-0x3F and IMSIC files 0x70-0xFF; anything else is an
        illegal instruction (5.2's recommendation, Spike) — a virtual one
        from VS-mode with Smaia (AIA 2.3).  Spike's order
        (sscsrind_reg_csr_t, then aia_ireg_proxy_csr_t)."""
        from .simrv import ILLEGAL, RVTrap
        m, v = self.m, self.m.v
        k = _IREG_INDEX[num & 0xFF]
        if num >> 8 == 3:                       # mireg*
            if _ireg_target(self, k, "m") is None:
                raise RVTrap(ILLEGAL, insn)
            return
        if num >> 8 == 1:                       # sireg*: 5.5's state enable
            _PRE_STATEEN_CSRIND(self, num, write, insn)
        if v and m.priv == 0:
            raise RVTrap(VIRTUAL_INSTRUCTION, insn)
        vs = num >> 8 == 2 or v                 # (vsireg*, or sireg* that is vsireg*)
        sel = self.vsiselect if vs else self.siselect
        if self.cdeleg and DELEG_FIRST <= sel <= DELEG_LAST:
            if num >> 8 == 2:                   # vsireg* from M / HS: illegal (9.1)
                if not v:
                    raise RVTrap(ILLEGAL, insn)
                raise RVTrap(VIRTUAL_INSTRUCTION if self.menvcfg & CDE else ILLEGAL, insn)
            if not self.menvcfg & CDE:
                raise RVTrap(ILLEGAL, insn)
            if v:
                raise RVTrap(VIRTUAL_INSTRUCTION, insn)
            if not self.mcounteren >> (sel - DELEG_FIRST) & 1:
                raise RVTrap(ILLEGAL, insn)
            if _deleg_target(self, sel, k) is not None:
                return
        a = self.aia
        if a is None or _aia_target(self, k, "vs" if vs else "s", sel) is None:
            raise RVTrap(VIRTUAL_INSTRUCTION if a is not None and a.smaia and v and m.priv == 1
                         else ILLEGAL, insn)
        imsic = 0x70 <= sel <= 0xFF
        if not vs and m.priv < 3 and a.mvien >> SEI & 1 and imsic:
            raise RVTrap(ILLEGAL, insn)         # (AIA 5.3: mvien.SEIP)
        if self.stateen:
            if not v and m.priv < 3 and imsic and not self.mstateen[0] & SE_IMSIC:
                raise RVTrap(ILLEGAL, insn)
            if v and imsic and not self.hstateen[0] & SE_IMSIC:
                raise RVTrap(VIRTUAL_INSTRUCTION, insn)
            if not vs and m.priv < 3 and sel < 0x40 and not self.mstateen[0] & SE_AIA:
                raise RVTrap(ILLEGAL, insn)     # (AIA 2.5: the S-level iprio array)

    def _alias(self, num: int) -> int:
        """The CSR an access of ``num`` reaches (V = 1: a VS twin)."""
        return VS_ALIAS.get(num, num) if self.m.v else num

    def read(self, num: int) -> int:
        return self.table[self._alias(num)].read(self) & self.m.amask

    def peek(self, num: int) -> int:
        """CSR ``num`` itself, whatever V (the views: with V = 1 an
        instruction naming sstatus reaches vsstatus, a debugger asking
        for sstatus sees sstatus)."""
        return self.table[num].read(self) & self.m.amask

    def read_for_write(self, num: int) -> int:
        """The value CSRRS / CSRRC modify (mip: the software-writable
        SEIP, not the external line — priv. 3.1.9)."""
        num = self._alias(num)
        if num in (0x344, 0x144):
            v = (self.mip() & ~(1 << SEI)) | (self.mip_sw & (1 << SEI))
            if num == 0x344:
                return v
            v &= self.mideleg & self.s_interrupts
            return self.aia.sip(v) & self.m.amask if self.aia is not None else v
        return self.read(num)

    def write(self, num: int, value: int) -> None:
        if num == 0x180 and self.m.v:           # (21.2.18: an unsupported MODE through
            _w_satp_virtual(self, value & self.m.amask)     # satp: no write)
            return
        w = self.table[self._alias(num)].write
        if w is not None:
            w(self, value & self.m.amask)

    # -- the counters (priv. 3.1.10 / Zicntr) -------------------------------------------

    def tick(self, cycles: int, mode: tuple[int, int] | None = None,
             skip_cycle: bool = False) -> None:
        """``cycles`` clock cycles in ``mode`` (priv, V; None: the hart's
        now): mcycle unless mcountinhibit.CY or mcyclecfg inhibits the mode
        (Smcntrpmf 7.3), the generic hart's event counters."""
        mode = (self.m.priv, self.m.v) if mode is None else mode
        if not skip_cycle and not self.mcountinhibit & 1 \
                and not self.mcyclecfg & _inh(*mode):
            self.mcycle = (self.mcycle + cycles) & M64
        if self.hpm is not None and self.hpm.armed:
            self.hpm.tick(cycles, mode)

    def retire(self, mode: tuple[int, int] | None = None) -> None:
        """An instruction retired; ``mode``: the one it started in (an xRET
        counts in the mode it returns from: 7.3)."""
        mode = (self.m.priv, self.m.v) if mode is None else mode
        if not self.mcountinhibit & 4 and not self.minstretcfg & _inh(*mode):
            self.minstret = (self.minstret + 1) & M64

    def misa(self) -> int:
        mxl = 1 if self.m.xlen == 32 else 2
        bits = 0
        exts = self.m.extensions
        for e in exts:
            if len(e) == 1 and e.isalpha():
                bits |= 1 << (ord(e) - ord("a"))
        rv32 = self.m.xlen == 32                 # unpriv. 28.4: MISA.C from Zca & co.
        if "zca" in exts and ("f" not in exts or (
                "d" not in exts and rv32 and "zcf" in exts) or (
                "d" in exts and "zcd" in exts and ("zcf" in exts or not rv32))):
            bits |= 1 << 2
        else:
            bits &= ~(1 << 2)
        if self.has_u:
            bits |= 1 << (ord("u") - ord("a"))
        if self.has_s:
            bits |= 1 << (ord("s") - ord("a"))
        return mxl << (self.m.xlen - 2) | bits


# ------------------------------------------------------------------ the table

def _always(c: CSRFile) -> bool:
    return True


def _rv32(c: CSRFile) -> bool:
    return c.m.xlen == 32


def _s(c: CSRFile) -> bool:
    return c.has_s


def _u(c: CSRFile) -> bool:
    return c.has_u


def _f(c: CSRFile) -> bool:
    return c.m.flen > 0 or getattr(c.m, "inx", False)     # (Zfinx: fcsr too, ch. 26)


def _seed(c: CSRFile) -> int:
    """seed's value: OPST = ES16 and 16 bits of a SHA-256 DRBG keyed by
    the hart and the poll count (replay-deterministic; unpriv. 31.5.2.3)."""
    import hashlib
    d = hashlib.sha256(f"emips seed {c.m.hartid} {c.seed_polls}".encode()).digest()
    return 0b10 << 30 | int.from_bytes(d[:2], "little")


def _attr(name: str) -> Callable[[CSRFile], int]:
    return lambda c: getattr(c, name)


def _set(name: str, mask: int | Callable[[CSRFile], int] = M64) -> Callable[[CSRFile, int], None]:
    def w(c: CSRFile, v: int) -> None:
        mk = mask(c) if callable(mask) else mask
        setattr(c, name, (getattr(c, name) & ~mk) | (v & mk))
    return w


def _counter(name: str, half: str) -> Callable[[CSRFile], int]:
    def read(c: CSRFile) -> int:
        v = c.m.mtime if name == "time" else getattr(c, name) if name else 0
        if name == "time" and c.m.v:            # 21.2.7: VS / VU read time + htimedelta
            v = (v + c.htimedelta) & M64
        if c.m.xlen == 32:
            return (v >> 32 if half == "h" else v) & M32
        return v
    return read


def _w_counter(name: str, half: str) -> Callable[[CSRFile, int], None]:
    def write(c: CSRFile, v: int) -> None:
        old = getattr(c, name)
        if c.m.xlen == 32:
            new = (old & M32) | (v & M32) << 32 if half == "h" else (old & ~M32) | (v & M32)
        else:
            new = v
        setattr(c, name, new & M64)
        c.m.counter_written.add(name)           # (takes effect after this instruction)
    return write


def _w_fcsr(part: str) -> Callable[[CSRFile, int], None]:
    def w(c: CSRFile, v: int) -> None:
        m = c.m
        if part == "fflags":
            m.fcsr = (m.fcsr & ~0x1F) | (v & 0x1F)
        elif part == "frm":
            m.fcsr = (m.fcsr & 0x1F) | (v & 7) << 5
        else:
            m.fcsr = v & 0xFF
        if m.flen:
            c.fs_dirty()                        # (FP state written: Dirty; Zfinx: no FS)
    return w


def _v(c: CSRFile) -> bool:
    return c.m.vector is not None


def _w_vector(name: str) -> Callable[[CSRFile, int], None]:
    """vstart (only the bits of the largest element index: VLEN - 1, as
    Spike), vxsat (bit 0), vxrm (1:0), vcsr (2:0) — 30.3.7-30.3.10; a write
    makes VS Dirty."""
    def w(c: CSRFile, v: int) -> None:
        vu = c.m.vector
        if name == "vstart":
            vu.vstart = v & (vu.vlen - 1)
        elif name == "vxsat":
            vu.vxsat = v & 1
        elif name == "vxrm":
            vu.vxrm = v & 3
        else:
            vu.vxsat, vu.vxrm = v & 1, v >> 1 & 3
        c.vs_dirty()
    return w


#: the vector CSRs (30.3, Table 43): VS = Off makes them illegal
VECTOR_CSRS = frozenset({0x008, 0x009, 0x00A, 0x00F, 0xC20, 0xC21, 0xC22})


def _w_tvec(name: str) -> Callable[[CSRFile, int], None]:
    return lambda c, v: setattr(c, name, v & ~2)     # MODE 2 / 3 reserved (Spike: ~2)


def _r_epc(name: str) -> Callable[[CSRFile], int]:
    return lambda c: getattr(c, name) & ~(3 if c.m.ialign == 4 else 1)


def _w_mip(c: CSRFile, v: int) -> None:
    mk = c.mip_mask()
    if c.aia is not None and c.aia.mvien >> SEI & 1:
        mk &= ~(1 << SEI)                       # (AIA 5.3: mip.SEIP read-only then)
    c.mip_sw = (c.mip_sw & ~mk) | (v & mk)


def _w_sip(c: CSRFile, v: int, half: int = M64) -> None:
    mk = ((1 << SSI) | c.mip_extra) & c.mideleg & half      # 10.1.3: only SSIP (and LCOFIP)
    if c.aia is not None:
        mk &= ~c.aia.sip_write(v, half)         # (AIA 5.3: the virtual bits are mvip's)
    c.mip_sw = (c.mip_sw & ~mk) | (v & mk)


def _w_sie(c: CSRFile, v: int, half: int = M64) -> None:
    mk = c.s_interrupts & c.mideleg & half
    if c.aia is not None:
        mk &= ~c.aia.sie_write(v, half)
    c.mie = (c.mie & ~mk) | (v & mk)


def _r_sip(c: CSRFile) -> int:
    v = c.mip() & c.mideleg & c.s_interrupts
    return c.aia.sip(v) if c.aia is not None else v


def _r_sie(c: CSRFile) -> int:
    v = c.mie & c.mideleg & c.s_interrupts
    return c.aia.sie(v) if c.aia is not None else v


def _w_satp(c: CSRFile, v: int) -> None:
    """priv. 10.1.11: a write of an unsupported MODE has no effect."""
    m = c.m
    mode = v >> 31 if m.xlen == 32 else v >> 60
    if mode in m.satp_modes:
        c.satp = v


def _h(c: CSRFile) -> bool:
    return c.has_h


def _h32(c: CSRFile) -> bool:
    return c.has_h and c.m.xlen == 32


def _satp_mode(c: CSRFile, v: int) -> int:
    return v >> 31 if c.m.xlen == 32 else v >> 60


def _w_satp_virtual(c: CSRFile, v: int) -> None:
    """21.2.18: with V = 1 a satp write with an unsupported MODE is
    ignored (no write to vsatp)."""
    if _satp_mode(c, v) in c.m.satp_modes:
        c.vsatp = v


def _w_vsatp(c: CSRFile, v: int) -> None:
    """vsatp itself (V = 0): WARL — an unsupported MODE keeps the old
    MODE, ASID and PPN are written (Spike's base_atp_csr_t)."""
    if _satp_mode(c, v) in c.m.satp_modes:
        c.vsatp = v
    else:
        mode = (1 << 31) if c.m.xlen == 32 else 0xF << 60
        c.vsatp = (c.vsatp & mode) | (v & ~mode)


def hgatp_modes(m) -> frozenset:
    """hgatp's MODEs (21.2.10, Table 41): Bare and the x4 twin of each
    satp mode (Sv32x4 = 1; Sv39x4 / Sv48x4 / Sv57x4 = 8-10)."""
    return m.satp_modes


def _w_hgatp(c: CSRFile, v: int) -> None:
    """21.2.10: PPN (16 KB aligned: PPN 1:0 read 0), VMID (VMIDLEN =
    VMIDMAX), MODE — an unsupported MODE keeps the old one (RV64; Spike)."""
    if c.m.xlen == 32:
        mask = 1 << 31 | 0x7F << 22 | 0x3FFFFF
    else:
        mask = 0x3FFF << 44 | ((1 << 44) - 1)
        if v >> 60 in hgatp_modes(c.m):
            mask |= 0xF << 60
    mask &= ~3
    c.hgatp = (c.hgatp & ~mask) | (v & mask)


def _w_hvip(c: CSRFile, v: int, half: int = M64) -> None:
    """21.2.3: VSEIP / VSTIP / VSSIP writable (VSSIP = mip.VSSIP); with the
    AIA bits 13-63 (6.3.2: hvien's virtual interrupts)."""
    if half & 1:
        c.mip_sw = (c.mip_sw & ~(1 << VSSI)) | (v & (1 << VSSI))
        c.hvip = v & ((1 << VSEI) | (1 << VSTI))
    a = c.aia
    if a is not None:
        from .rvaia import HIGH
        mk = HIGH & half
        a.hvip_hi = (a.hvip_hi & ~mk) | (v & mk)


def _r_hvip(c: CSRFile) -> int:
    v = c.hvip | (c.mip_sw & (1 << VSSI))
    return v | c.aia.hvip_hi if c.aia is not None else v


def _w_hip(c: CSRFile, v: int) -> None:
    mk = (1 << VSSI) & c.mideleg                # (only VSSIP: 21.2.3)
    c.mip_sw = (c.mip_sw & ~mk) | (v & mk)


def _w_hie(c: CSRFile, v: int) -> None:
    mk = HS_INTERRUPTS & c.mideleg
    c.mie = (c.mie & ~mk) | (v & mk)


def _vs_mask(c: CSRFile) -> int:
    return VS_INTERRUPTS & c.hideleg & c.mideleg


def _w_vsip(c: CSRFile, v: int, half: int = M64) -> None:
    """21.2.12: vsip.SSIP = hip.VSSIP when hideleg delegates it; with the
    AIA bits 13-63 are hvip's where hvien has them (Table 6.1)."""
    if half & 1:
        mk = (1 << VSSI) & _vs_mask(c)
        c.mip_sw = (c.mip_sw & ~mk) | ((v << 1) & mk)
    a = c.aia
    if a is not None:
        mk = a.hvien & half
        a.hvip_hi = (a.hvip_hi & ~mk) | (v & mk)


def _w_vsie(c: CSRFile, v: int, half: int = M64) -> None:
    if half & 1:
        mk = _vs_mask(c)
        c.mie = (c.mie & ~mk) | ((v << 1) & mk)
    a = c.aia
    if a is not None:
        mk = a.hvien & half
        a.vsie_v = (a.vsie_v & ~mk) | (v & mk)


def _r_vsip(c: CSRFile) -> int:
    v = (c.mip() & _vs_mask(c)) >> 1
    return v | c.aia.vsip_hi() if c.aia is not None else v


def _r_vsie(c: CSRFile) -> int:
    v = (c.mie & _vs_mask(c)) >> 1
    return v | c.aia.vsie_hi() if c.aia is not None else v


def _w_vsstatus(c: CSRFile, v: int) -> None:
    mk = c.sstatus_mask()
    c.vsstatus = (c.vsstatus & ~mk) | (v & mk)


def _r_vsstatus(c: CSRFile) -> int:
    read = c.sstatus_mask() | (3 << XS_SHIFT) | (1 << UBE) | (1 << (c.m.xlen - 1))
    if c.m.xlen == 64:
        read |= 3 << UXL_SHIFT
    return c.vsstatus_value() & read


def _context_bits(c: CSRFile) -> int:
    """mcontext / hcontext: 6 / 13 bits, 7 / 14 with H (Debug spec 5.7.9;
    Spike)."""
    return (1 << ((6 if c.m.xlen == 32 else 13) + (1 if c.has_h else 0))) - 1


def _w_dcsr(c: CSRFile, v: int) -> None:
    """Debug spec 4.9.1: ebreakm / ebreaks / ebreaku (with S / U), step,
    mprven and prv (an unsupported mode: the least privileged, as MPP)
    are writable; stepie, stopcount, stoptime read 0 (Spike); debugver,
    cause read-only."""
    from . import rvdebug as D
    mk = D.EBREAKM | D.STEP | D.MPRVEN | (D.EBREAKS if c.has_s else 0) \
        | (D.EBREAKU if c.has_u else 0) \
        | (D.EBREAKVS | D.EBREAKVU | D.DCSR_V if c.has_h else 0) \
        | (DCSR_PELP if c.zicfilp else 0)       # (4.9.1: H; pelp with Zicfilp)
    c.dcsr = (c.dcsr & ~(mk | 3)) | (v & mk) | c.legal_mode(v & 3)


def _trig(c: CSRFile) -> bool:
    return c.m.tm.count > 0


def _envcfg_mask(c: CSRFile, level: str) -> int:
    """The writable bits of menvcfg / henvcfg / senvcfg (3.1.18, 21.2.5,
    10.1.4): FIOM (with S: may be read-only 0 without it), the CMOs' CBIE /
    CBCFE / CBZE (unpriv. 19.5), and at M / H level ADUE (Svadu), PBMTE
    (Svpbmt), STCE (Sstc), at M CDE (Smcdeleg) — Spike's masks."""
    e = c.m.extensions
    mask = FIOM if c.has_s else 0
    if "zicbom" in e:
        mask |= CBIE | CBCFE
    if "zicboz" in e:
        mask |= CBZE
    if level != "s":
        if "svadu" in e:
            mask |= ADUE
        if "svpbmt" in e:
            mask |= PBMTE
        if c.sstc:
            mask |= STCE
    if level == "m" and c.cdeleg:
        mask |= CDE
    if c.zicfilp:                               # (LF10: 3.1.18, 10.1.4, 21.2.5)
        mask |= LPE
    if c.zicfiss:
        mask |= SSE
    return mask


def _w_envcfg(name: str, level: str, half: str) -> Callable[[CSRFile, int], None]:
    """xenvcfg's write: CBIE's reserved 10 becomes 00 (Spike); henvcfg's
    PBMTE / STCE / ADUE only while menvcfg's are set; STCE turned off
    leaves STIP / VSTIP as the comparators last had them (Spike's
    ``time_counter_csr_t::sync``)."""
    base = _w64(name, lambda c: _envcfg_mask(c, level), half)

    def w(c: CSRFile, v: int) -> None:
        rv32h = half == "h" and c.m.xlen == 32
        v64 = (v & M32) << 32 if rv32h else v
        if not rv32h and (v >> 4) & 3 == 2:
            v64 &= ~CBIE
        if level == "h":
            keep = c.menvcfg | ~(PBMTE | STCE | ADUE | SSE)
            v64 = (v64 & keep) | (c.henvcfg & ~keep)
        if level == "s":                        # (SSE kept while it reads 0: Spike)
            gate = c.henv() if c.m.v else c.menvcfg
            keep = gate | ~SSE
            v64 = (v64 & keep) | (c.senvcfg & ~keep)
        v = v64 >> 32 if rv32h else v64
        old_m, old_h = c.menvcfg & STCE, c.henv() & STCE
        if c.sstc and (old_m or old_h):         # (the comparators' last signals)
            sig = c.timer_signals()
        base(c, v)
        if c.sstc and old_m and not c.menvcfg & STCE:
            c.mip_sw = (c.mip_sw & ~(1 << STI)) | sig & (1 << STI)
        if c.sstc and old_h and not c.henv() & STCE:
            c.mip_sw = (c.mip_sw & ~(1 << VSTI)) | sig & (1 << VSTI)
    return w


# -- Smstateen / Ssstateen (4.1, 4.2; Spike's masks) ---------------------------------

def _stateen(c: CSRFile) -> bool:
    return c.stateen


def _mstateen_mask(c: CSRFile, i: int) -> int:
    """mstateen0: C, FCSR (Zfinx), JVT (Zcmt), with S ENVCFG and CONTEXT,
    CSRIND (Sscsrind), P1P13 (RV32 with H), SE0; mstateen1-3: SE0."""
    if i:
        return SE0
    return _hstateen_mask(c, 0) | (SE_P1P13 if c.has_h and c.m.xlen == 32 else 0)


def _hstateen_mask(c: CSRFile, i: int) -> int:
    if i:
        return SE0
    return _sstateen_mask(c, 0) | SE0 | (SE_ENVCFG | SE_CONTEXT if c.has_s else 0) \
        | (SE_CSRIND if c.sscsrind else 0) | (SE_AIA if c.ssaia else 0) \
        | (SE_IMSIC if c.ssaia and c.imsic() is not None else 0)     # (AIA 2.5)


def _sstateen_mask(c: CSRFile, i: int) -> int:
    if i:
        return 0
    e = c.m.extensions
    return SE_C | (SE_FCSR if "zfinx" in e else 0) | (SE_JVT if "zcmt" in e else 0)


def _w_mstateen(i: int, half: str) -> Callable[[CSRFile, int], None]:
    def w(c: CSRFile, v: int) -> None:
        mk = _mstateen_mask(c, i)
        if c.m.xlen == 32:
            mk &= (M32 << 32) if half == "h" else M32
            v = (v & M32) << 32 if half == "h" else v & M32
        c.mstateen[i] = (c.mstateen[i] & ~mk) | (v & mk)
    return w


def _r_hstateen(i: int) -> Callable[[CSRFile], int]:
    """4.1: a bit mstateen holds 0 reads 0 in hstateen."""
    return lambda c: c.hstateen[i] & c.mstateen[i]


def _w_hstateen(i: int, half: str) -> Callable[[CSRFile, int], None]:
    def w(c: CSRFile, v: int) -> None:
        mk = _hstateen_mask(c, i) & c.mstateen[i]
        if c.m.xlen == 32:
            mk &= (M32 << 32) if half == "h" else M32
            v = (v & M32) << 32 if half == "h" else v & M32
        c.hstateen[i] = (c.hstateen[i] & ~mk) | (v & mk)
    return w


def _r_sstateen(i: int) -> Callable[[CSRFile], int]:
    """4.1: mstateen's zeros (and, accessed in VS-mode, hstateen's) read 0."""
    def r(c: CSRFile) -> int:
        v = c.sstateen[i] & c.mstateen[i]
        return v & c.hstateen[i] if c.m.v else v
    return r


def _w_sstateen(i: int) -> Callable[[CSRFile, int], None]:
    def w(c: CSRFile, v: int) -> None:
        mk = _sstateen_mask(c, i) & c.mstateen[i]
        if c.m.v:
            mk &= c.hstateen[i]
        c.sstateen[i] = (c.sstateen[i] & ~mk) | (v & mk)
    return w


# -- the checks Spike makes before the privilege ---------------------------------------

def _pre_stateen(bit: int, *, virt: bool = True, index: int | None = None):
    """mstateen (below M: illegal), then with V = 1 hstateen (virtual)."""
    def check(c: CSRFile, num: int, write: bool, insn: int) -> None:
        if not c.stateen:
            return
        from .simrv import ILLEGAL, RVTrap
        i = (num & 3) if index is None else index
        if c.m.priv < 3 and not c.mstateen[i] & bit:
            raise RVTrap(ILLEGAL, insn)
        if virt and c.m.v and not c.hstateen[i] & bit:
            raise RVTrap(VIRTUAL_INSTRUCTION, insn)
    return check


def _pre_timecmp(c: CSRFile, num: int, write: bool, insn: int) -> None:
    """Sstc 19.1.4 / 19.3, Spike's order: menvcfg.STCE (below M), time's
    counter enables, henvcfg.STCE (V = 1)."""
    from .simrv import ILLEGAL, RVTrap
    if not c.menvcfg & STCE and c.m.priv < 3:
        raise RVTrap(ILLEGAL, insn)
    c.counter_check(1, insn)
    if c.m.v and not c.henv() & STCE:
        raise RVTrap(VIRTUAL_INSTRUCTION, insn)


_PRE_CHECKS: dict[int, Callable] = {
    0x011: lambda c, num, write, insn: c.sse_check(insn),      # (ssp: 22.2.1)
    0x10A: _pre_stateen(SE_ENVCFG, index=0),                    # senvcfg
    0x60A: _pre_stateen(SE_ENVCFG, virt=False, index=0),        # henvcfg
    0x61A: _pre_stateen(SE_ENVCFG, virt=False, index=0),
    0x612: _pre_stateen(SE_P1P13, virt=False, index=0),         # hedelegh
    0x5A8: _pre_stateen(SE_CONTEXT, index=0),                   # scontext / hcontext
    0x6A8: _pre_stateen(SE_CONTEXT, index=0),
    **{n: _pre_stateen(SE0, virt=False) for n in (*range(0x60C, 0x610), *range(0x61C, 0x620))},
    **{n: _pre_stateen(SE0) for n in range(0x10C, 0x110)},     # sstateen0-3
    **{n: _pre_stateen(SE_CSRIND, index=0) for n in (0x150, 0x250, 0x251, 0x252,
                                                      0x253, 0x255, 0x256, 0x257)},
    # (sireg*: the privilege first, then the state enables — Spike's
    # virtualized_indirect_csr_t; CSRFile.indirect_check)
    **{n: _pre_timecmp for n in (0x14D, 0x15D, 0x24D, 0x25D)},
    # the AIA (2.5; Spike's aia_csr_t / aia_rv32_high_csr_t / *stopi / *topei)
    **{n: _pre_stateen(SE_AIA, index=0) for n in (0x114, 0x154, 0xDB0, 0x613, 0x608, 0x618,
                                                  0x655, 0x609, 0x646, 0x656, 0x647, 0x657,
                                                  0x214, 0x254, 0xEB0, 0x313, 0x314, 0x354)},
    0x15C: lambda c, num, write, insn: _pre_stopei(c, num, write, insn),
    0x25C: _pre_stateen(SE_IMSIC, index=0),
}


def _pre_stopei(c: CSRFile, num: int, write: bool, insn: int) -> None:
    """stopei (AIA 2.5 / 5.3): the IMSIC state enable, then — S-mode, V = 0
    — mvien.SEIP makes the S file the machine level's (illegal)."""
    _pre_stateen(SE_IMSIC, index=0)(c, num, write, insn)
    if c.m.priv < 3 and not c.m.v and c.aia.mvien >> SEI & 1:
        from .simrv import ILLEGAL, RVTrap
        raise RVTrap(ILLEGAL, insn)


_PRE_STATEEN_CSRIND = _pre_stateen(SE_CSRIND, index=0)


# -- Smcsrind / Sscsrind and Smcdeleg (5, 9) -----------------------------------------

#: the alias CSRs' number low byte -> i of *iregi (mireg = 1 ... ; 0x54 is miph)
_IREG_INDEX = {0x51: 1, 0x52: 2, 0x53: 3, 0x55: 4, 0x56: 5, 0x57: 6}
_INDIRECT = frozenset({base + k for base in (0x300, 0x100, 0x200) for k in _IREG_INDEX})


def _csrind(c: CSRFile) -> bool:
    return c.csrind


def _sscsrind(c: CSRFile) -> bool:
    return c.sscsrind


def _deleg_target(c: CSRFile, sel: int, k: int):
    """Smcdeleg's Table 20: (read, write) of the state sireg``k`` reaches
    while siselect = ``sel`` (0x40 + counter), or None (illegal: sireg3 /
    sireg6, sireg4 / sireg5 on RV64, 0x41 = time, state the hart lacks —
    9.1)."""
    n = sel - DELEG_FIRST
    rv32 = c.m.xlen == 32
    if k in (3, 6) or (k in (4, 5) and not rv32) or n == 1:
        return None
    hi = k in (4, 5)
    if k in (1, 4):                             # the counter
        if n == 0:
            name = "mcycle"
        elif n == 2:
            name = "minstret"
        elif c.hpm is not None:
            hpm = c.hpm
            mask = (M32 << 32) if hi else (M32 if rv32 else M64)
            def write(v: int) -> None:
                hpm.write_count(n, (v & M32) << 32 if hi else v, mask)
                if c.m.count_mode is not None:          # (3.1.10: after this instruction)
                    c.m.counter_written.add(f"hpm{n}")
            return (lambda: (hpm.count[n] >> 32 if hi else hpm.count[n] & mask) & M64, write)
        else:
            return None
        rd, wr = _counter(name, "h" if hi else ""), _w_counter(name, "h" if hi else "")
        return lambda: rd(c), lambda v: wr(c, v)
    # the event / configuration (sireg2 / sireg5): MINH reads 0 and stays (9.1)
    if n in (0, 2):
        if not c.smcntrpmf:
            return None
        name = "mcyclecfg" if n == 0 else "minstretcfg"
        get, mask = (lambda: getattr(c, name)), _cfg_mask(c)
    elif c.hpm is not None and (not hi or c.sscofpmf):
        hpm = c.hpm
        get, mask = (lambda: hpm.event[n]), hpm.event_mask()
    else:
        return None
    mask &= ~(1 << 62)
    if rv32:
        mask &= (M32 << 32) if hi else M32

    def read() -> int:
        v = get() & ~(1 << 62)
        return (v >> 32) & M32 if hi else v & c.m.amask

    def write(v: int) -> None:
        v = (v & M32) << 32 if hi else v
        new = (get() & ~mask) | (v & mask)
        if n in (0, 2):
            setattr(c, name, new)
        else:
            c.hpm.write_event(n, new)
    return read, write


def _ireg_target(c: CSRFile, k: int, level: str):
    """The register *ireg``k`` of ``level`` ("m", "s", "vs") reaches now:
    (read, write) or None — Smcdeleg's counters (Table 20), the AIA's iprio
    array (0x30-0x3F) and IMSIC file registers (0x70-0xFF; AIA 2.1-2.3)."""
    sel = {"m": c.miselect, "s": c.siselect, "vs": c.vsiselect}[level]
    if level == "s" and c.cdeleg and DELEG_FIRST <= sel <= DELEG_LAST:
        return _deleg_target(c, sel, k)
    return _aia_target(c, k, level, sel)


def _aia_target(c: CSRFile, k: int, level: str, sel: int):
    a = c.aia
    if a is None or k != 1:
        return None
    if 0x30 <= sel <= 0x3F and level != "vs" and (level == "s" or a.smaia):
        return a.iprio_reg(level, sel, c.m.xlen)
    if 0x70 <= sel <= 0xFF:
        im = c.imsic()
        f = None if im is None else im.file(level)
        return f.reg(sel, c.m.xlen) if f is not None else None
    return None


def _ireg(k: int, level: str) -> Callable[[CSRFile], int]:
    def read(c: CSRFile) -> int:
        t = _ireg_target(c, k, level)
        return t[0]() if t is not None else 0
    return read


def _w_ireg(k: int, level: str) -> Callable[[CSRFile, int], None]:
    def write(c: CSRFile, v: int) -> None:
        t = _ireg_target(c, k, level)
        if t is not None:
            t[1](v)
    return write


def _w_iselect(name: str) -> Callable[[CSRFile, int], None]:
    """*iselect (5.2-5.4): WARL with at least 0..0xFFF — eMIPS holds every
    bit, as Spike (a select it does not implement makes *ireg* illegal)."""
    return lambda c, v: setattr(c, name, v & c.m.amask)


def _cfg_mask(c: CSRFile) -> int:
    """mcyclecfg / minstretcfg (7.2.1): the xINH bits of the hart's modes."""
    from .rvhpm import inh_mask
    return inh_mask(c)


# -- the counters with Zihpm / Sscofpmf (3.1.10, 20) ---------------------------------

def _r_hpm(n: int, half: str) -> Callable[[CSRFile], int]:
    def r(c: CSRFile) -> int:
        if c.hpm is None:
            return 0
        v = c.hpm.count[n]
        if c.m.xlen == 32:
            return (v >> 32 if half == "h" else v) & M32
        return v
    return r


def _w_hpm(n: int, half: str) -> Callable[[CSRFile, int], None]:
    def w(c: CSRFile, v: int) -> None:
        if c.hpm is None:
            return
        if c.m.xlen == 32:
            mk = (M32 << 32) if half == "h" else M32
            v = (v & M32) << 32 if half == "h" else v & M32
        else:
            mk = M64
        c.hpm.write_count(n, v, mk)
        if c.m.count_mode is not None:          # (an instruction's: it takes effect after
            c.m.counter_written.add(f"hpm{n}")  # that instruction, 3.1.10)
    return w


def _r_event(n: int, half: str) -> Callable[[CSRFile], int]:
    def r(c: CSRFile) -> int:
        if c.hpm is None:
            return 0
        v = c.hpm.event[n]
        if c.m.xlen == 32:
            return (v >> 32 if half == "h" else v) & M32
        return v
    return r


def _w_event(n: int, half: str) -> Callable[[CSRFile, int], None]:
    def w(c: CSRFile, v: int) -> None:
        if c.hpm is None:
            return
        if c.m.xlen == 32:
            mk = (M32 << 32) if half == "h" else M32
            v = (v & M32) << 32 if half == "h" else v & M32
        else:
            mk = M64
        c.hpm.write_event(n, v, mk)
    return w


def _scountovf(c: CSRFile) -> int:
    """20.2: the OF bits — below M only those mcounteren (and with V = 1
    hcounteren) allow."""
    v = c.hpm.overflowed() if c.hpm is not None else 0
    if c.m.priv < 3:
        v &= c.mcounteren
    if c.m.v:
        v &= c.hcounteren
    return v


def _w_scountinhibit(c: CSRFile, v: int) -> None:
    """9.2: the delegated counters' mcountinhibit bits (mcounteren's);
    the others stay (Spike clears them)."""
    mk = c.mcounteren & c.counter_mask() & ~2
    c.mcountinhibit = (c.mcountinhibit & ~mk) | (v & mk)


# -- Smrnmi (8.3) ----------------------------------------------------------------------

def _w_mnstatus(c: CSRFile, v: int) -> None:
    """NMIE can be set but not cleared; MNPP legalised as MPP; MNPV with H."""
    mpp = c.legal_mode((v >> MNPP_SHIFT) & 3)
    keep = c.mnstatus & NMIE
    v = (v & (NMIE | (MNPV if c.has_h else 0) | (MNPELP if c.zicfilp else 0))) \
        | mpp << MNPP_SHIFT
    c.mnstatus = keep | v


# -- Smepmp's mseccfg (6.2; Zkr's SSEED / USEED: 31.5.3) ------------------------------

def _w_mseccfg(c: CSRFile, v: int) -> None:
    """MMWP / MML sticky, RLB locked at 0 once a PMP entry has L (6.2.2b;
    Spike's mseccfg_csr_t), SSEED / USEED with Zkr."""
    m = c.m
    new = c.mseccfg
    if c.smepmp and m.pmp.entries:
        locked = any(m.pmp.cfg[i] & 0x80 for i in range(m.pmp.entries))
        if not (locked and not new & RLB):
            new = (new & ~RLB) | (v & RLB)
        new |= v & (MMWP | MML)
        m.pmp.version += 1
    if "zkr" in m.extensions:
        mk = (USEED if c.has_u else 0) | (SSEED if c.has_s else 0)
        new = (new & ~mk) | (v & mk)
    if c.zicfilp:                               # (3.1.19: MLPE)
        new = (new & ~MLPE) | (v & MLPE)
    c.mseccfg = new
    m.pmp.mseccfg = new


def _half(base: Callable[[CSRFile], int], half: str) -> Callable[[CSRFile], int]:
    return (lambda c: base(c) >> 32 & M32) if half == "h" else (lambda c: base(c) & M32 if
                                                               c.m.xlen == 32 else base(c))


def _w64(name: str, mask: Callable[[CSRFile], int], half: str) -> Callable[[CSRFile, int], None]:
    """A 64-bit register's write (RV32: one half)."""
    def w(c: CSRFile, v: int) -> None:
        mk = mask(c)
        if c.m.xlen == 32:
            mk &= (M32 << 32) if half == "h" else M32
            v = (v & M32) << 32 if half == "h" else v & M32
        setattr(c, name, (getattr(c, name) & ~mk) | (v & mk))
    return w


# -- the AIA (AIA chs. 2, 3, 5, 6; emips.rvaia) ---------------------------------------

def _wlo(w: Callable) -> Callable[[CSRFile, int], None]:
    """A 64-bit interrupt register's low CSR: RV32 writes bits 31:0 only."""
    return lambda c, v: w(c, v, M32 if c.m.xlen == 32 else M64)


def _whi(w: Callable) -> Callable[[CSRFile, int], None]:
    """Its RV32 high half (AIA 2.1: miph, sieh ...): bits 63:32."""
    return lambda c, v: w(c, (v & M32) << 32, M32 << 32)


def _hi(r: Callable[[CSRFile], int]) -> Callable[[CSRFile], int]:
    return lambda c: r(c) >> 32 & M32


def _smaia(c: CSRFile) -> bool:
    return c.smaia


def _smaia_s(c: CSRFile) -> bool:
    return c.smaia and c.has_s


def _ssaia(c: CSRFile) -> bool:
    return c.ssaia


def _ssaia_h(c: CSRFile) -> bool:
    return c.ssaia and c.has_h


def _rv32_and(f: Callable[[CSRFile], bool]) -> Callable[[CSRFile], bool]:
    return lambda c: f(c) and c.m.xlen == 32


def _with_imsic(f: Callable[[CSRFile], bool]) -> Callable[[CSRFile], bool]:
    return lambda c: f(c) and c.imsic() is not None


def _hgeie_mask(c: CSRFile) -> int:
    im = c.imsic()
    return sum(1 << g for g in im.guests) if im is not None else 0


def _w_hstatus(c: CSRFile, v: int) -> None:
    """21.2.1; with Ssaia VGEIN (AIA 3.7): WLRL — a value naming no guest
    interrupt file keeps the old one (Spike)."""
    c.hstatus = (c.hstatus & ~HSTATUS_MASK) | (v & HSTATUS_MASK)
    if c.ssaia:
        g = (v & VGEIN_MASK) >> VGEIN_SHIFT
        im = c.imsic()
        if g == 0 or (im is not None and g in im.guests):
            c.hstatus = (c.hstatus & ~VGEIN_MASK) | g << VGEIN_SHIFT


def _w_mvien(c: CSRFile, v: int, half: int = M64) -> None:
    a = c.aia
    mk = a.mvien_mask & half
    a.mvien = (a.mvien & ~mk) | (v & mk)        # (5.3: mvip.SSIP's own bit unspecified
    #                                             after 0 -> 1: the last one, as Spike)


def _w_hvien(c: CSRFile, v: int, half: int = M64) -> None:
    from .rvaia import HIGH
    a = c.aia
    mk = HIGH & half
    a.hvien = (a.hvien & ~mk) | (v & mk)


def _w_hviprio(i: int) -> Callable[[CSRFile, int, int], None]:
    from .rvaia import HVIPRIO

    def w(c: CSRFile, v: int, half: int = M64) -> None:
        a = c.aia
        mk = sum(0xFF << (8 * b) for (r, b) in HVIPRIO if r == i) & half
        a.hviprio[i] = (a.hviprio[i] & ~mk) | (v & mk)
    return w


def _w_hvictl(c: CSRFile, v: int) -> None:
    from .rvaia import HVICTL_MASK
    c.aia.hvictl = v & HVICTL_MASK


def _topei(level: str) -> Callable[[CSRFile], int]:
    """*topei (AIA 3.9): bits 26:16 and 10:0 the file's top identity."""
    def r(c: CSRFile) -> int:
        f = c.imsic().file(level)
        t = f.topei() if f is not None else 0
        return t << 16 | t
    return r


def _w_topei(level: str) -> Callable[[CSRFile, int], None]:
    def w(c: CSRFile, v: int) -> None:
        f = c.imsic().file(level)
        if f is not None:
            f.claim()                           # (the value written is ignored)
    return w


CSRS: list[CSR] = [
    # -- unprivileged (Zicsr / F 20.2 / Zicntr / Zihpm) --
    CSR(0x001, "fflags", _f, lambda c: c.m.fcsr & 0x1F, _w_fcsr("fflags")),
    CSR(0x002, "frm", _f, lambda c: c.m.fcsr >> 5 & 7, _w_fcsr("frm")),
    CSR(0x003, "fcsr", _f, lambda c: c.m.fcsr & 0xFF, _w_fcsr("fcsr")),
    # V (30.3, Table 43)
    CSR(0x008, "vstart", _v, lambda c: c.m.vector.vstart, _w_vector("vstart")),
    CSR(0x009, "vxsat", _v, lambda c: c.m.vector.vxsat, _w_vector("vxsat")),
    CSR(0x00A, "vxrm", _v, lambda c: c.m.vector.vxrm, _w_vector("vxrm")),
    CSR(0x00F, "vcsr", _v, lambda c: c.m.vector.vcsr, _w_vector("vcsr")),
    CSR(0xC20, "vl", _v, lambda c: c.m.vector.vl),
    CSR(0xC21, "vtype", _v, lambda c: c.m.vector.vtype),
    CSR(0xC22, "vlenb", _v, lambda c: c.m.vector.vlenb),
    # Zcmt (unpriv. 28.14.3): BASE XLEN-1:6 (WARL), MODE 5:0 WARL = 0 (jump
    # table mode, the only one)
    # Zicfiss (33.2.2): ssp, URW, bits 1:0 (RV64 2:0: UXLEN is never 32) read-only 0
    CSR(0x011, "ssp", lambda c: c.zicfiss, _attr("ssp"),
        lambda c, v: setattr(c, "ssp", v & c.m.amask & ~(c.m.xlen // 8 - 1))),
    CSR(0x017, "jvt", lambda c: "zcmt" in c.m.extensions, _attr("jvt"),
        lambda c, v: setattr(c, "jvt", v & c.m.amask & ~0x3F)),
    # Zkr (unpriv. 31.5): seed — ES16 + 16 bits from a DRBG (a "virtual
    # source", 31.5.2.3), a new value after each poll (the write); mseccfg
    # (priv. 3.1.17) with SSEED / USEED (31.5.3)
    CSR(0x015, "seed", lambda c: "zkr" in c.m.extensions, lambda c: _seed(c),
        lambda c, v: setattr(c, "seed_polls", c.seed_polls + 1)),
    CSR(0x747, "mseccfg", lambda c: "zkr" in c.m.extensions or c.smepmp or c.zicfilp,
        lambda c: c.mseccfg & c.m.amask, _w_mseccfg),
    CSR(0x757, "mseccfgh", lambda c: ("zkr" in c.m.extensions or c.smepmp or c.zicfilp)
        and c.m.xlen == 32,
        lambda c: 0, lambda c, v: None),
    CSR(0xC00, "cycle", _always, _counter("mcycle", "")),
    CSR(0xC01, "time", _always, _counter("time", "")),
    CSR(0xC02, "instret", _always, _counter("minstret", "")),
    CSR(0xC80, "cycleh", _rv32, _counter("mcycle", "h")),
    CSR(0xC81, "timeh", _rv32, _counter("time", "h")),
    CSR(0xC82, "instreth", _rv32, _counter("minstret", "h")),
    *(CSR(0xC00 + n, f"hpmcounter{n}", lambda c: c.zihpm, _r_hpm(n, "")) for n in range(3, 32)),
    *(CSR(0xC80 + n, f"hpmcounter{n}h", lambda c: c.zihpm and c.m.xlen == 32, _r_hpm(n, "h"))
      for n in range(3, 32)),
    # Sscofpmf (20.2)
    CSR(0xDA0, "scountovf", lambda c: c.sscofpmf and c.has_s, _scountovf),
    # -- supervisor (priv. 10.1) --
    CSR(0x100, "sstatus", _s, CSRFile.sstatus_value,
        lambda c, v: c.write_mstatus(v, c.sstatus_mask())),
    CSR(0x104, "sie", _s, _r_sie, _wlo(_w_sie)),
    CSR(0x105, "stvec", _s, _attr("stvec"), _w_tvec("stvec")),
    CSR(0x106, "scounteren", _s, _attr("scounteren"), _set("scounteren", CSRFile.counter_mask)),
    CSR(0x10A, "senvcfg", _s, lambda c: c.senv(), _w_envcfg("senvcfg", "s", "")),
    # Smcdeleg / Ssccfg (9.2)
    CSR(0x120, "scountinhibit", lambda c: c.cdeleg, lambda c: c.mcounteren & c.mcountinhibit,
        _w_scountinhibit),
    # Sstc (19.1.1)
    CSR(0x14D, "stimecmp", lambda c: c.sstc, _half(_attr("stimecmp"), ""),
        _w64("stimecmp", lambda c: M64, "")),
    CSR(0x15D, "stimecmph", lambda c: c.sstc and c.m.xlen == 32, lambda c: c.stimecmp >> 32,
        _w64("stimecmp", lambda c: M64, "h")),
    # Smstateen / Ssstateen (4.2)
    *(CSR(0x10C + i, f"sstateen{i}", lambda c: c.stateen and c.has_s, _r_sstateen(i),
          _w_sstateen(i)) for i in range(4)),
    # Sscsrind (5.3)
    CSR(0x150, "siselect", _sscsrind, _attr("siselect"), _w_iselect("siselect")),
    *(CSR(0x100 + n, f"sireg{'' if k == 1 else k}", _sscsrind, _ireg(k, "s"), _w_ireg(k, "s"))
      for n, k in _IREG_INDEX.items()),
    CSR(0x140, "sscratch", _s, _attr("sscratch"), _set("sscratch")),
    CSR(0x141, "sepc", _s, _r_epc("sepc"), lambda c, v: setattr(c, "sepc", v & ~1)),
    CSR(0x142, "scause", _s, _attr("scause"), _set("scause")),
    CSR(0x143, "stval", _s, _attr("stval"), _set("stval")),
    CSR(0x144, "sip", _s, _r_sip, _wlo(_w_sip)),
    CSR(0x180, "satp", _s, _attr("satp"), _w_satp),
    # -- the hypervisor extension (priv. ch. 21) --
    CSR(0x600, "hstatus", _h, _attr("hstatus"), _w_hstatus),
    CSR(0x602, "hedeleg", _h, lambda c: c.hedeleg & c.m.amask,
        _w64("hedeleg", CSRFile.hedeleg_mask, "")),
    CSR(0x612, "hedelegh", _h32, lambda c: c.hedeleg >> 32,
        _w64("hedeleg", CSRFile.hedeleg_mask, "h")),
    CSR(0x603, "hideleg", _h, lambda c: c.hideleg & c.mideleg,
        _set("hideleg", VS_INTERRUPTS)),
    CSR(0x604, "hie", _h, lambda c: c.mie & HS_INTERRUPTS & c.mideleg, _w_hie),
    CSR(0x605, "htimedelta", _h, _half(_attr("htimedelta"), ""),
        _w64("htimedelta", lambda c: M64, "")),
    CSR(0x615, "htimedeltah", _h32, lambda c: c.htimedelta >> 32,
        _w64("htimedelta", lambda c: M64, "h")),
    CSR(0x606, "hcounteren", _h, _attr("hcounteren"), _set("hcounteren", CSRFile.counter_mask)),
    CSR(0x607, "hgeie", _h, _attr("hgeie"), _set("hgeie", _hgeie_mask)),
    CSR(0xE12, "hgeip", _h, lambda c: c.imsic().hgeip if c.imsic() is not None else 0),
    CSR(0x60A, "henvcfg", _h, _half(lambda c: c.henv(), ""), _w_envcfg("henvcfg", "h", "")),
    CSR(0x61A, "henvcfgh", _h32, lambda c: c.henv() >> 32, _w_envcfg("henvcfg", "h", "h")),
    *(CSR(0x60C + i, f"hstateen{i}", lambda c: c.stateen and c.has_h,
          _half(_r_hstateen(i), ""), _w_hstateen(i, "")) for i in range(4)),
    *(CSR(0x61C + i, f"hstateen{i}h", lambda c: c.stateen and c.has_h and c.m.xlen == 32,
          lambda c, i=i: _r_hstateen(i)(c) >> 32, _w_hstateen(i, "h")) for i in range(4)),
    CSR(0x24D, "vstimecmp", lambda c: c.sstc and c.has_h, _half(_attr("vstimecmp"), ""),
        _w64("vstimecmp", lambda c: M64, "")),
    CSR(0x25D, "vstimecmph", lambda c: c.sstc and c.has_h and c.m.xlen == 32,
        lambda c: c.vstimecmp >> 32, _w64("vstimecmp", lambda c: M64, "h")),
    CSR(0x250, "vsiselect", lambda c: c.sscsrind and c.has_h, _attr("vsiselect"),
        _w_iselect("vsiselect")),
    *(CSR(0x200 + n, f"vsireg{'' if k == 1 else k}", lambda c: c.sscsrind and c.has_h,
          _ireg(k, "vs"), _w_ireg(k, "vs")) for n, k in _IREG_INDEX.items()),
    CSR(0x643, "htval", _h, _attr("htval"), _set("htval")),
    CSR(0x644, "hip", _h, lambda c: c.mip() & HS_INTERRUPTS & c.mideleg, _w_hip),
    CSR(0x645, "hvip", _h, _r_hvip, _wlo(_w_hvip)),
    CSR(0x64A, "htinst", _h, _attr("htinst"), _set("htinst")),
    CSR(0x680, "hgatp", _h, _attr("hgatp"), _w_hgatp),
    CSR(0x200, "vsstatus", _h, _r_vsstatus, _w_vsstatus),
    CSR(0x204, "vsie", _h, _r_vsie, _wlo(_w_vsie)),
    CSR(0x205, "vstvec", _h, _attr("vstvec"), _w_tvec("vstvec")),
    CSR(0x240, "vsscratch", _h, _attr("vsscratch"), _set("vsscratch")),
    CSR(0x241, "vsepc", _h, _r_epc("vsepc"), lambda c, v: setattr(c, "vsepc", v & ~1)),
    CSR(0x242, "vscause", _h, _attr("vscause"), _set("vscause")),
    CSR(0x243, "vstval", _h, _attr("vstval"), _set("vstval")),
    CSR(0x244, "vsip", _h, _r_vsip, _wlo(_w_vsip)),
    CSR(0x280, "vsatp", _h, _attr("vsatp"), _w_vsatp),
    CSR(0x34A, "mtinst", _h, _attr("mtinst"), _set("mtinst")),
    CSR(0x34B, "mtval2", _h, _attr("mtval2"), _set("mtval2")),
    # -- machine (priv. 3.1) --
    CSR(0xF11, "mvendorid", _always, lambda c: 0),
    CSR(0xF12, "marchid", _always, lambda c: 0),
    CSR(0xF13, "mimpid", _always, lambda c: 0),
    CSR(0xF14, "mhartid", _always, lambda c: c.m.hartid),
    CSR(0xF15, "mconfigptr", _always, lambda c: 0),
    CSR(0x300, "mstatus", _always, CSRFile.mstatus_value,
        lambda c, v: c.write_mstatus(v if c.m.xlen == 64 else (c.mstatus & ~M32) | v)),
    CSR(0x301, "misa", _always, CSRFile.misa, lambda c, v: None),
    CSR(0x302, "medeleg", _s, lambda c: c.medeleg & (M32 if c.m.xlen == 32 else M64),
        _w64("medeleg", CSRFile.medeleg_mask, "")),
    CSR(0x303, "mideleg", _s, _attr("mideleg"), _set("mideleg", lambda c: c.s_interrupts)),
    CSR(0x304, "mie", _always, _attr("mie"), _set("mie", CSRFile.mie_mask)),
    CSR(0x305, "mtvec", _always, _attr("mtvec"), _w_tvec("mtvec")),
    CSR(0x306, "mcounteren", _u, _attr("mcounteren"), _set("mcounteren", CSRFile.counter_mask)),
    CSR(0x30A, "menvcfg", _u, _half(_attr("menvcfg"), ""), _w_envcfg("menvcfg", "m", "")),
    *(CSR(0x30C + i, f"mstateen{i}", _stateen, _half(lambda c, i=i: c.mstateen[i], ""),
          _w_mstateen(i, "")) for i in range(4)),
    *(CSR(0x31C + i, f"mstateen{i}h", lambda c: c.stateen and c.m.xlen == 32,
          lambda c, i=i: c.mstateen[i] >> 32, _w_mstateen(i, "h")) for i in range(4)),
    # Smcntrpmf (7.2.1)
    CSR(0x321, "mcyclecfg", lambda c: c.smcntrpmf, _half(_attr("mcyclecfg"), ""),
        _w64("mcyclecfg", _cfg_mask, "")),
    CSR(0x322, "minstretcfg", lambda c: c.smcntrpmf, _half(_attr("minstretcfg"), ""),
        _w64("minstretcfg", _cfg_mask, "")),
    CSR(0x721, "mcyclecfgh", lambda c: c.smcntrpmf and c.m.xlen == 32,
        lambda c: c.mcyclecfg >> 32, _w64("mcyclecfg", _cfg_mask, "h")),
    CSR(0x722, "minstretcfgh", lambda c: c.smcntrpmf and c.m.xlen == 32,
        lambda c: c.minstretcfg >> 32, _w64("minstretcfg", _cfg_mask, "h")),
    # Smcsrind (5.2)
    CSR(0x350, "miselect", _csrind, _attr("miselect"), _w_iselect("miselect")),
    *(CSR(0x300 + n, f"mireg{'' if k == 1 else k}", _csrind, _ireg(k, "m"), _w_ireg(k, "m"))
      for n, k in _IREG_INDEX.items()),
    # Smaia (AIA 2.1): the machine level
    CSR(0x35C, "mtopei", _with_imsic(_smaia), _topei("m"), _w_topei("m")),
    CSR(0xFB0, "mtopi", _smaia, lambda c: c.aia.mtopi()),
    CSR(0x308, "mvien", _smaia_s, lambda c: c.aia.mvien, _wlo(_w_mvien)),
    CSR(0x309, "mvip", _smaia_s, lambda c: c.aia.mvip_read(),
        _wlo(lambda c, v, h: c.aia.mvip_write(v, h))),
    CSR(0x313, "midelegh", _rv32_and(_smaia_s), lambda c: c.mideleg >> 32, lambda c, v: None),
    CSR(0x314, "mieh", _rv32_and(_smaia), lambda c: c.mie >> 32, lambda c, v: None),
    CSR(0x318, "mvienh", _rv32_and(_smaia_s), lambda c: c.aia.mvien >> 32, _whi(_w_mvien)),
    CSR(0x319, "mviph", _rv32_and(_smaia_s), lambda c: c.aia.mvip_read() >> 32,
        _whi(lambda c, v, h: c.aia.mvip_write(v, h))),
    CSR(0x354, "miph", _rv32_and(_smaia), lambda c: c.mip() >> 32, lambda c, v: None),
    # Ssaia (AIA 2.2 / 2.3): the supervisor and VS levels
    CSR(0x15C, "stopei", _with_imsic(_ssaia), _topei("s"), _w_topei("s")),
    CSR(0xDB0, "stopi", _ssaia, lambda c: c.aia.stopi()),
    CSR(0x114, "sieh", _rv32_and(_ssaia), _hi(_r_sie), _whi(_w_sie)),
    CSR(0x154, "siph", _rv32_and(_ssaia), _hi(_r_sip), _whi(_w_sip)),
    CSR(0x608, "hvien", _ssaia_h, lambda c: c.aia.hvien, _wlo(_w_hvien)),
    CSR(0x609, "hvictl", _ssaia_h, lambda c: c.aia.hvictl, _w_hvictl),
    CSR(0x646, "hviprio1", _ssaia_h, lambda c: c.aia.hviprio[0], _wlo(_w_hviprio(0))),
    CSR(0x647, "hviprio2", _ssaia_h, lambda c: c.aia.hviprio[1], _wlo(_w_hviprio(1))),
    CSR(0x25C, "vstopei", _with_imsic(_ssaia_h), _topei("vs"), _w_topei("vs")),
    CSR(0xEB0, "vstopi", _ssaia_h, lambda c: c.aia.vstopi()),
    CSR(0x613, "hidelegh", _rv32_and(_ssaia_h), lambda c: (c.hideleg & c.mideleg) >> 32,
        lambda c, v: None),
    CSR(0x618, "hvienh", _rv32_and(_ssaia_h), lambda c: c.aia.hvien >> 32, _whi(_w_hvien)),
    CSR(0x655, "hviph", _rv32_and(_ssaia_h), _hi(_r_hvip), _whi(_w_hvip)),
    CSR(0x656, "hviprio1h", _rv32_and(_ssaia_h), lambda c: c.aia.hviprio[0] >> 32,
        _whi(_w_hviprio(0))),
    CSR(0x657, "hviprio2h", _rv32_and(_ssaia_h), lambda c: c.aia.hviprio[1] >> 32,
        _whi(_w_hviprio(1))),
    CSR(0x214, "vsieh", _rv32_and(_ssaia_h), _hi(_r_vsie), _whi(_w_vsie)),
    CSR(0x254, "vsiph", _rv32_and(_ssaia_h), _hi(_r_vsip), _whi(_w_vsip)),
    # Smrnmi (8.3)
    CSR(0x740, "mnscratch", lambda c: c.smrnmi, _attr("mnscratch"), _set("mnscratch")),
    CSR(0x741, "mnepc", lambda c: c.smrnmi, _r_epc("mnepc"),
        lambda c, v: setattr(c, "mnepc", v & ~1)),
    CSR(0x742, "mncause", lambda c: c.smrnmi, _attr("mncause"), lambda c, v: None),
    CSR(0x744, "mnstatus", lambda c: c.smrnmi, _attr("mnstatus"), _w_mnstatus),
    CSR(0x310, "mstatush", _rv32, lambda c: c.mstatus >> 32 & M32,
        lambda c, v: c.write_mstatus((c.mstatus & M32) | (v & M32) << 32,
                                     c.mstatus_mask() & ~M32)),     # (21.4.1: MPV / GVA)
    CSR(0x312, "medelegh", lambda c: c.has_s and c.m.xlen == 32, lambda c: c.medeleg >> 32,
        _w64("medeleg", CSRFile.medeleg_mask, "h")),
    CSR(0x31A, "menvcfgh", lambda c: c.has_u and c.m.xlen == 32, lambda c: c.menvcfg >> 32,
        _w_envcfg("menvcfg", "m", "h")),
    CSR(0x320, "mcountinhibit", _always, _attr("mcountinhibit"),
        _set("mcountinhibit", lambda c: c.counter_mask() & ~2)),
    *(CSR(0x320 + n, f"mhpmevent{n}", _always, _r_event(n, ""), _w_event(n, ""))
      for n in range(3, 32)),
    *(CSR(0x720 + n, f"mhpmevent{n}h", lambda c: c.sscofpmf and c.m.xlen == 32,
          _r_event(n, "h"), _w_event(n, "h")) for n in range(3, 32)),
    CSR(0x340, "mscratch", _always, _attr("mscratch"), _set("mscratch")),
    CSR(0x341, "mepc", _always, _r_epc("mepc"), lambda c, v: setattr(c, "mepc", v & ~1)),
    CSR(0x342, "mcause", _always, _attr("mcause"), _set("mcause")),
    CSR(0x343, "mtval", _always, _attr("mtval"), _set("mtval")),
    CSR(0x344, "mip", _always, CSRFile.mip, _w_mip),
    CSR(0xB00, "mcycle", _always, _counter("mcycle", ""), _w_counter("mcycle", "")),
    CSR(0xB02, "minstret", _always, _counter("minstret", ""), _w_counter("minstret", "")),
    CSR(0xB80, "mcycleh", _rv32, _counter("mcycle", "h"), _w_counter("mcycle", "h")),
    CSR(0xB82, "minstreth", _rv32, _counter("minstret", "h"), _w_counter("minstret", "h")),
    # -- PMP (3.7.1): every pmpaddr / pmpcfg exists; RV64 has only the even pmpcfgs --
    *(CSR(0x3A0 + n, f"pmpcfg{n}", lambda c, n=n: c.m.pmp.cfg_exists(n),
          lambda c, n=n: c.m.pmp.read_cfg(n), lambda c, v, n=n: c.m.pmp.write_cfg(n, v))
      for n in range(16)),
    *(CSR(0x3B0 + i, f"pmpaddr{i}", _always, lambda c, i=i: c.m.pmp.read_addr(i),
          lambda c, v, i=i: c.m.pmp.write_addr(i, v)) for i in range(64)),
    *(CSR(0xB00 + n, f"mhpmcounter{n}", _always, _r_hpm(n, ""), _w_hpm(n, ""))
      for n in range(3, 32)),
    # -- the trigger module (Debug spec 5.7; emips.rvdebug) --
    CSR(0x7A0, "tselect", _trig, lambda c: c.m.tm.tselect, lambda c, v: c.m.tm.write_tselect(v)),
    CSR(0x7A1, "tdata1", _trig, lambda c: c.m.tm.read1(), lambda c, v: c.m.tm.write1(v)),
    CSR(0x7A2, "tdata2", _trig, lambda c: c.m.tm.read2(), lambda c, v: c.m.tm.write2(v)),
    CSR(0x7A3, "tdata3", _trig, lambda c: c.m.tm.read3(), lambda c, v: c.m.tm.write3(v)),
    CSR(0x7A4, "tinfo", _trig, lambda c: TINFO, lambda c, v: None),
    CSR(0x7A5, "tcontrol", lambda c: _trig(c) and not c.has_s, lambda c: c.m.tm.tcontrol,
        lambda c, v: setattr(c.m.tm, "tcontrol", v & (MTE | MPTE))),
    CSR(0x7A8, "mcontext", _trig, lambda c: c.m.tm.mcontext,
        lambda c, v: setattr(c.m.tm, "mcontext", v & _context_bits(c))),
    CSR(0x6A8, "hcontext", lambda c: _trig(c) and c.has_h, lambda c: c.m.tm.mcontext,
        lambda c, v: setattr(c.m.tm, "mcontext", v & _context_bits(c))),     # (5.7.7: alias)
    CSR(0x5A8, "scontext", lambda c: _trig(c) and c.has_s, lambda c: c.m.tm.scontext,
        lambda c, v: setattr(c.m.tm, "scontext", v & ((1 << (16 if c.m.xlen == 32 else 32)) - 1))),
    # -- Debug Mode (Debug spec 4.9; Sdext) --
    CSR(0x7B0, "dcsr", _always, _attr("dcsr"), _w_dcsr),
    CSR(0x7B1, "dpc", _always, _r_epc("dpc"), lambda c, v: setattr(c, "dpc", v & ~1)),
    CSR(0x7B2, "dscratch0", _always, _attr("dscratch0"), _set("dscratch0")),
    CSR(0x7B3, "dscratch1", _always, _attr("dscratch1"), _set("dscratch1")),
    *(CSR(0xB80 + n, f"mhpmcounter{n}h", _rv32, _r_hpm(n, "h"), _w_hpm(n, "h"))
      for n in range(3, 32)),
]
NAMES = {c.num: c.name for c in CSRS}
